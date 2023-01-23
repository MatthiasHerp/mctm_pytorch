import torch

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

    nf_mctm = NF_MCTM(polynomial_range=torch.tensor([[-15], [15]]), number_variables=3, spline_decorrelation="bspline")

    #nf_mctm.forward(y, inverse=False)

    train(nf_mctm, y, iterations=200, penalty_params=torch.FloatTensor([0,0,0]), verbose=False)
    plt.show()

    output_tuple = nf_mctm.forward(y, inverse=False)
    z = output_tuple[0].detach().numpy()

    sns.kdeplot(x=z[:, 0], y=z[:, 1])
    plt.show()
    sns.kdeplot(x=z[:, 0], y=z[:, 2])
    plt.show()
    sns.kdeplot(x=z[:, 1], y=z[:, 2])
    plt.show()