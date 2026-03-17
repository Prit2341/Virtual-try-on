import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.warp_model import WarpNet
from model.warp_utils import warp_cloth


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "dataset/train/tensors"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs/warp"

BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4


class TensorDataset(Dataset):

    def __init__(self, path):
        self.files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".pt")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = torch.load(self.files[idx])

        person = data["person"]
        cloth = data["cloth"]
        cloth_mask = data["cloth_mask"]
        pose = data["pose_map"]
        agnostic = data["agnostic"]

        person_inputs = torch.cat([agnostic, pose], 0)

        return person_inputs, cloth, cloth_mask


def save_checkpoint(model, epoch):

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    path = os.path.join(
        CHECKPOINT_DIR,
        f"warp_epoch_{epoch:03d}.pth"
    )

    torch.save(model.state_dict(), path)


def train():

    dataset = TensorDataset(DATA_PATH)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = WarpNet().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    l1 = nn.L1Loss()

    writer = SummaryWriter(LOG_DIR)

    step = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        for person_inputs, cloth, mask in loader:

            person_inputs = person_inputs.to(DEVICE)
            cloth = cloth.to(DEVICE)
            mask = mask.unsqueeze(1).to(DEVICE)

            flow = model(person_inputs)

            warped = warp_cloth(cloth, flow)

            loss = l1(warped * mask, cloth * mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar(
                "warp_loss",
                loss.item(),
                step
            )

            step += 1

        print(
            f"Epoch {epoch} | Loss {loss.item():.4f}"
        )

        save_checkpoint(model, epoch)


if __name__ == "__main__":
    train()