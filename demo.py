import torch

from bernstein_transformation_layer import *
from training_helpers import *


if __name__ == '__main__':

    y_distribution = Laplace(5, 3)
    y = y_distribution.sample((2000,2)) # Generate training data
    #plt.hist(y[:,0].numpy(), bins=100)
    #plt.hist(y[:,1].numpy(), bins=100)
    #plt.show()

    polynomial_range = torch.FloatTensor([[-30, -30],
                                          [40, 40]])
    mctm = Transformation(degree=10, number_variables=2, polynomial_range=polynomial_range)

    train(mctm, y, iterations=20, verbose=False)

    z, log_d = mctm.forward(y)
    plt.show()

    plt.hist(z[:, 0].detach().numpy(), bins=100)
    plt.hist(z[:, 1].detach().numpy(), bins=100)
    plt.show()

    polynomial_range = torch.FloatTensor([[-10, -10],
                                          [10, 10]])
    mctm.approximate_inverse(y, polynomial_range_inverse=polynomial_range, iterations=20)

    evaluate(mctm)
    plt.show()