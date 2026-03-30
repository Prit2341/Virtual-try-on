import torch
from viton_model import VITON

model = VITON().cuda()

person = torch.randn(1, 25, 256, 192).cuda()
cloth = torch.randn(1, 3, 256, 192).cuda()

output, warped = model(person, cloth)

print(output.shape)