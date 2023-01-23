import torch
from torch import nn

class Flip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.flip(x, [0])