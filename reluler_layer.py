import torch
from torch import nn

class ReLULeR(): #nn.Module
    def __init__(self,polynomial_range_abs):
        #super().__init__()
        self.polynomial_range_abs = polynomial_range_abs
        self.l0 = nn.ReLU()
        self.l1 = nn.ReLU()

    def forward(self,x):
        #x element [-polynomial_range_abs, polynomial_range_abs]
        #x_0_1_theory element [0, 1] in theory
        x_0_1_theory = (x + self.polynomial_range_abs) / (self.polynomial_range_abs * 2)
        # x_0_1_sure element [0, 1] for sure
        x_0_1_sure = -1*(self.l1(-1*self.l0(x_0_1_theory)+1)-1)
        # x_05_05_sure element [-0.5, 0.5] for sure
        x_05_05_sure = x_0_1_sure - 0.5
        # x_well_behaved element [-polynomial_range_abs, polynomial_range_abs] for sure
        x_well_behaved = x_05_05_sure * 2 * self.polynomial_range_abs

        return x_well_behaved

if __name__ == "__main__":
    reluler = ReLULeR(5)
    x = torch.tensor([[-5,-10,0,3,4,8],[-4,-2,1,3,2,12]]).T
    print(torch.allclose(reluler(x),torch.FloatTensor([[-5,-5,0,3,4,5],[-4,-2,1,3,2,5]]).T,0.1))