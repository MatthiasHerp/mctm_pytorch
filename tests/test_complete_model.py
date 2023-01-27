import torch

from training_helpers import *
from nf_mctm import *


if __name__ == '__main__':

    # Reproducibility
    # Infos from here: https://pytorch.org/docs/stable/notes/randomness.html
    # Set Seeds for Torch, Numpy and Python
    torch.manual_seed(1)
    import numpy as np
    np.random.seed(1)
    import random
    random.seed(1)


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

    nf_mctm = NF_MCTM(input_min=y.min(0).values,
                      input_max=y.max(0).values,
                      polynomial_range=torch.tensor([[-5], [5]]),
                      number_variables=3,
                      spline_decorrelation="bspline")

    #normalisation = Normalisation(input_min=y.min(0).values, input_max=y.max(0).values, output_range=torch.tensor([5]))
    #y_norm = (y - y.min(0).values) / (y.max(0).values - y.min(0).values) - 0.5
    #y_norm = y_norm * 2 * torch.FloatTensor([4])
    ##y_norm = y #normalisation(y)
#
    #if y_norm.isnan().sum() > 0:
    #    print("y_norm contains NaNs")
    #    print(y_norm)
    #    exit()


    loss = train(nf_mctm, y, iterations=200, verbose=False)
    plt.show()

    log_likelihood = nf_mctm.log_likelihood(y=y)

    z = nf_mctm.latent_space_representation(y).detach().numpy()

    sns.kdeplot(x=z[:, 0], y=z[:, 1])
    plt.show()
    sns.kdeplot(x=z[:, 0], y=z[:, 2])
    plt.show()
    sns.kdeplot(x=z[:, 1], y=z[:, 2])
    plt.show()