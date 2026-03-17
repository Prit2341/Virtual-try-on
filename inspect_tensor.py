import torch
import os

path = "dataset/train/tensors"

file = os.listdir(path)[0]

data = torch.load(os.path.join(path, file))

print(data.keys())