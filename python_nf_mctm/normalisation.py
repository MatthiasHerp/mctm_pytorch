import torch
from torch import nn

class Normalisation(nn.Module): #nn.Module
    def __init__(self,input_min,input_max,output_range):
        super().__init__()
        self.input_min = input_min
        self.input_max = input_max
        self.output_range = output_range

    def forward(self,x):
        x = (x - self.input_min) / (self.input_max - self.input_min) - 0.5
        x = x * 2 * self.output_range
        return x

if __name__ == '__main__':
    x = torch.tensor([1,2,3,4,5,6,7,8,9,10])
    input_min = torch.min(x)
    input_max = torch.max(x)
    output_range = 2
    normalisation_layer = Normalisation(input_min,input_max,output_range)
    print(normalisation_layer(x))
    x2 = torch.tensor([-1, 2, 3, 4, 5, 6, 7, 8, 9, 12])
    print(normalisation_layer(x2))

    loc = torch.zeros(3)
    lam = torch.Tensor([[1, 0, 0],
                        [3, 1, 0],
                        [0, 0, 1]])
    scale = lam @ torch.eye(3) @ torch.transpose(lam, 0, 1)
    from torch.distributions.multivariate_normal import MultivariateNormal
    y_distribution = MultivariateNormal(loc, scale)
    y = y_distribution.sample((2000, 1))  # Generate training data
    y = y.reshape((2000, 3))

    normalisation_layer = Normalisation(y.min(0).values, y.max(0).values, output_range)
    z = normalisation_layer(y)
    print(z.min(0).values)
    print(z.max(0).values)

