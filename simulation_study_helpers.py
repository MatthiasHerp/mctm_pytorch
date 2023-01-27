import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import random
from scipy import stats
import time
from torch.distributions.multivariate_normal import MultivariateNormal
from pingouin import multivariate_normality
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

def set_seeds(seed_int):
    # Reproducibility
    # Infos from here: https://pytorch.org/docs/stable/notes/randomness.html
    # Set Seeds for Torch, Numpy and Python
    torch.manual_seed(seed_int)
    np.random.seed(seed_int)
    random.seed(seed_int)

# only for normal data
def create_data(sample_num=2000):
    loc = torch.zeros(3)
    lam = torch.Tensor([[1, 0, 0],
                        [3, 1, 0],
                        [0, 0, 1]])
    scale = lam @ torch.eye(3) @ torch.transpose(lam, 0, 1)
    y_distribution = MultivariateNormal(loc, scale)
    y = y_distribution.sample((sample_num, 1))  # Generate training data
    log_likelihood = y_distribution.log_prob(y)
    y = y.reshape((sample_num, 3))

    return y,log_likelihood

# only for normal data
def create_uniform_test_grid(num_observations=2000,ci_border=0.99):
    loc = torch.zeros(3)
    lam = torch.Tensor([[1, 0, 0],
                        [3, 1, 0],
                        [0, 0, 1]])

    y_i = torch.linspace(1.0-ci_border, ci_border,round(num_observations**(1/3)))
    y_1,y_2,y_3 = torch.meshgrid(y_i, y_i, y_i, indexing='ij')
    grid = torch.vstack([y_1.flatten(),
                         y_2.flatten(),
                         y_3.flatten()])

    uni_standard_normal = torch.distributions.Normal(0,1)

    obs_grid = uni_standard_normal.icdf(grid.flatten())

    obs_grid = obs_grid.reshape(grid.size())

    obs_grid = lam @ obs_grid

    obs_grid = obs_grid.T

    scale = lam @ torch.eye(3) @ torch.transpose(lam, 0, 1)
    y_distribution = MultivariateNormal(loc, scale)
    log_likelihood = y_distribution.log_prob(obs_grid)

    return obs_grid, log_likelihood

def test_kl_divergence(model,test_data, test_likelihood):
    z = model.forward(test_data,train=False)
    loc = torch.zeros(3)
    scale = torch.eye(3)
    y_distribution = MultivariateNormal(loc, scale)
    pred_log_likelihood = y_distribution.log_prob(z)
    #False need a fct that gives likelihood of the data given the network, require the log determinants

    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    #both input and target need to be in log likelihood
    return kl_loss(pred_log_likelihood, test_likelihood)

def evaluate_latent_space(z):
    # https://www.statology.org/multivariate-normality-test-r/
    # https://github.com/raphaelvallat/pingouin/blob/master/pingouin/multivariate.py
    # https://www.math.kit.edu/stoch/~henze/seite/aiwz/media/tests-mvn-test-2020.pdf
    res = multivariate_normality(z, alpha=.05)
    k2, p  = stats.normaltest(z)

    return res.normal, res.pval, np.mean(z,0), np.cov(z,rowvar=False), p


from itertools import combinations

def density_plots(data):
    num_cols = data.shape[1]
    fig, axs = plt.subplots(nrows=1, ncols=num_cols, figsize=(15,5),
                            gridspec_kw={'wspace':0.0, 'hspace':0.0},sharey=True)
    a=0
    for i, j in combinations(range(num_cols), 2):
        if i != j:
            sns.kdeplot(ax=axs[a], x=data[:, j], y=data[:, i])
            a+=1
    plt.subplots_adjust(wspace=0.05)

    return fig

def plot_latent_space(z):
    fig = density_plots(z)
    return fig