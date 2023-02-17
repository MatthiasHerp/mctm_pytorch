import torch
from torch import nn
import warnings

def adjust_ploynomial_range(polynomial_range: object, span_factor: object) -> object:
    span = polynomial_range[1] - polynomial_range[0]
    span_change = torch.tensor([-span_factor*span, span_factor*span], dtype=torch.float32)
    polynomial_range = polynomial_range + span_change
    return polynomial_range

class ReLULeR(): #nn.Module
    def __init__(self,polynomial_range: torch.Tensor):
        #super().__init__()
        self.polynomial_range = polynomial_range
        self.l0 = nn.ReLU()
        self.l1 = nn.ReLU()

    def forward(self,x: torch.Tensor):
        if torch.any(x < self.polynomial_range[0]):
            warnings.warn("Warning: x is smaller than polynomial_range[0], "
                          "maybe you should adjust the polynomial_range to be smaller")
        if torch.any(x > self.polynomial_range[1]):
            warnings.warn("Warning: x is bigger than polynomial_range[1], "
                          "maybe you should adjust the polynomial_range to be larger")

        #x element [-polynomial_range_abs, polynomial_range_abs]
        #x_0_1_theory element [0, 1] in theory
        x_0_1_theory = (x - self.polynomial_range[0]) / (self.polynomial_range[1] - self.polynomial_range[0])
        # x_0_1_sure element [0, 1] for sure
        x_0_1_sure = -1*(self.l1(-1*self.l0(x_0_1_theory)+1)-1)
        # x_well_behaved element [-polynomial_range_abs, polynomial_range_abs] for sure
        x_well_behaved = x_0_1_sure * (self.polynomial_range[1] - self.polynomial_range[0]) + self.polynomial_range[0]

        return x_well_behaved

def custom_sigmoid(input: torch.Tensor, polynomial_range: torch.Tensor):
    input_01 = (input - polynomial_range[0]) / (polynomial_range[1] - polynomial_range[0])
    input_11 = input_01 * 2 - 1
    input_bounded_01 = 1 / (1 + torch.exp(-input_11 * 4))
    input_bounded = input_bounded_01 * (polynomial_range[1] - polynomial_range[0]) + polynomial_range[0]

    return input_bounded

if __name__ == "__main__":
    reluler = ReLULeR(torch.tensor([-5,5]))
    x = torch.tensor([[-5,-10,0,3,4,8],[-4,-2,1,3,2,12]]).T
    print(torch.allclose(reluler.forward(x),torch.FloatTensor([[-5,-5,0,3,4,5],[-4,-2,1,3,2,5]]).T,0.1))