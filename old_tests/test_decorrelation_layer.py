import torch

from decorrelation_layer import *
from training_helpers import *
from nf_mctm import *


if __name__ == '__main__':
    from torch.distributions.multivariate_normal import MultivariateNormal

    loc = torch.zeros(3)
    lam = torch.Tensor([[1, 0, 0],
                        [3, 1, 0],
                        [0, 0, 1]])
    scale = lam @ torch.eye(3) @ torch.transpose(lam, 0, 1)
    y_distribution = MultivariateNormal(loc, scale)
    y = y_distribution.sample((2000, 1))  # Generate training data
    y = y.reshape((2000, 3))
    y.size()

    sns.kdeplot(x=y[:, 0], y=y[:, 1])
    plt.show()
    sns.kdeplot(x=y[:, 0], y=y[:, 2])
    plt.show()
    sns.kdeplot(x=y[:, 1], y=y[:, 2])
    plt.show()

    polynomial_range = torch.FloatTensor([[-15, -15, -15],
                                          [15, 15, 15]])
    decorr = Decorrelation(degree=12,
                           number_variables=3,
                           polynomial_range=polynomial_range,
                           spline="bspline",
                           calc_method="deBoor")

    train(decorr, y, iterations=200, verbose=False)
    plt.show()

    z = decorr.forward(y, train=False).detach().numpy()

    sns.kdeplot(x=z[:, 0], y=z[:, 1])
    plt.show()
    sns.kdeplot(x=z[:, 0], y=z[:, 2])
    plt.show()
    sns.kdeplot(x=z[:, 1], y=z[:, 2])
    plt.show()