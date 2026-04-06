from pathlib import Path
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class VITONDataset(Dataset):
    """Loads preprocessed VITON .pt tensors. Caches in RAM."""

    def __init__(self, root, max_samples=None, cache_in_ram=True):
        files = sorted(Path(root).glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files in {root}")
        self.files = files[:max_samples] if max_samples else files
        self.cache = None

        if cache_in_ram:
            print(f"Caching {len(self.files)} samples in RAM...", flush=True)
            self.cache = []
            for f in tqdm(self.files, desc="Loading dataset", unit="sample"):
                self.cache.append(torch.load(f, map_location="cpu", weights_only=False))
            print(f"Dataset cached in RAM ({len(self.cache)} samples)", flush=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.cache is not None:
            return self.cache[idx]
        return torch.load(self.files[idx], map_location="cpu", weights_only=False)


TENSOR_KEYS = ["person", "cloth", "agnostic", "pose_map", "cloth_mask", "parse_map"]


def make_loader(root, batch_size, max_samples=None, shuffle=True, num_workers=4):
    """Standard PyTorch DataLoader over VITONDataset (no caching, no custom threads).

    Returns batches as CPU tensors — caller must move to device with
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    """
    dataset = VITONDataset(root, max_samples=max_samples, cache_in_ram=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )


def _load_one(path):
    """Load a single .pt file from disk (called by thread pool)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _collate_batch(samples):
    """Stack a list of sample dicts into a single batch dict."""
    batch = {}
    for k in TENSOR_KEYS:
        batch[k] = torch.stack([s[k] for s in samples], dim=0)
    return batch


class FastVITONLoader:
    """Prefetch-queue data loader with parallel disk reads.

    Architecture:
      - N worker threads each prepare batches independently
      - Within each batch, a thread pool loads files in parallel (8 concurrent reads)
      - Loaded batches go into pinned CPU memory for fast async GPU transfer
      - Main thread pulls from the queue -- GPU never waits

    Pipeline:
      Worker 1: [load 128 files in parallel] -> [collate] -> [pin] -> queue
      Worker 2: [load 128 files in parallel] -> [collate] -> [pin] -> queue
      Worker 3: [load 128 files in parallel] -> [collate] -> [pin] -> queue
            ...
      Main:     queue.get() -> GPU transfer (non_blocking) -> train step

    Args:
        root:       path to .pt tensor directory
        batch_size: samples per batch
        device:     target device ("cuda")
        prefetch:   max batches buffered in queue (default 10)
        num_workers: parallel batch-preparation threads (default 4)
        io_threads:  parallel file reads within each batch (default 8)
    """

    def __init__(self, root, batch_size, device="cuda", max_samples=None,
                 prefetch=12, num_workers=6, io_threads=16):
        files = sorted(Path(root).glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files in {root}")
        self.files = files[:max_samples] if max_samples else files
        self.batch_size = batch_size
        self.device = device
        self.prefetch = prefetch
        self.num_workers = num_workers
        self.io_threads = io_threads

        self.n_samples = len(self.files)
        self.n_batches = self.n_samples // batch_size  # drop last

        print(f"FastVITONLoader: {self.n_samples} samples, "
              f"{self.n_batches} batches of {batch_size}, "
              f"prefetch={prefetch}, workers={num_workers}, "
              f"io_threads={io_threads}", flush=True)

    def __len__(self):
        return self.n_batches

    def _load_and_collate(self, indices):
        """Load one batch: read files in parallel, collate, pin memory."""
        files = self.files
        # Parallel disk reads using thread pool
        with ThreadPoolExecutor(max_workers=self.io_threads) as pool:
            samples = list(pool.map(_load_one, [files[i] for i in indices]))
        batch = _collate_batch(samples)
        # Pin memory for fast async CPU -> GPU transfer
        return {k: v.pin_memory() for k, v in batch.items()}

    def __iter__(self):
        # Shuffle file indices each epoch
        perm = torch.randperm(self.n_samples).tolist()
        batch_indices = []
        for i in range(self.n_batches):
            start = i * self.batch_size
            batch_indices.append(perm[start : start + self.batch_size])

        # Output queue: workers produce, main thread consumes
        out_q = queue.Queue(maxsize=self.prefetch)
        # Work queue: distributes batch jobs to workers
        work_q = queue.Queue()
        for bi in batch_indices:
            work_q.put(bi)

        def _worker():
            while True:
                try:
                    indices = work_q.get_nowait()
                except queue.Empty:
                    break
                batch = self._load_and_collate(indices)
                out_q.put(batch)

        # Launch worker threads
        workers = []
        for _ in range(self.num_workers):
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            workers.append(t)

        # Yield batches as they become available
        for _ in range(self.n_batches):
            batch = out_q.get()
            yield {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        for t in workers:
            t.join()
