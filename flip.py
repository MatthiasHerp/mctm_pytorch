import torch
from torch import nn

class Flip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.flip(x, [1])

if __name__ == "__main__":
    flipper = Flip()
    x = torch.tensor([[1,2,3,4,5],[-1,-2,-3,-4,-5]])
    print(x)
    print(flipper(x))