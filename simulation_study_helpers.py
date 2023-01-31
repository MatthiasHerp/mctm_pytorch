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
from bspline_prediction import *

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

def kl_divergence(target_log_likelihood, predicted_log_likelihood, mean=True):
    if mean:
        return torch.mean(torch.exp(target_log_likelihood) * (target_log_likelihood - predicted_log_likelihood))
    else:
        return torch.exp(target_log_likelihood) * (target_log_likelihood - predicted_log_likelihood)

def evaluate_latent_space(z):
    # https://www.statology.org/multivariate-normality-test-r/
    # https://github.com/raphaelvallat/pingouin/blob/master/pingouin/multivariate.py
    # https://www.math.kit.edu/stoch/~henze/seite/aiwz/media/tests-mvn-test-2020.pdf

    # Need to transpose as the test requires: rows=observations, columns=variables
    res = multivariate_normality(z, alpha=.05)
    res_normal = res.normal
    res_normality = res.pval
    k2, p  = stats.normaltest(z)

    return res_normal, res_normality, np.mean(z,0), np.cov(z,rowvar=False), p


from itertools import combinations

def plot_densities(data,x_lim=None,y_lim=None):
    num_cols = data.shape[1]
    num_combinations = int(num_cols * (num_cols - 1) / 2)

    if num_combinations > 1 :
        fig, axs = plt.subplots(nrows=1, ncols=num_combinations, figsize=(15,5),
                                gridspec_kw={'wspace':0.0, 'hspace':0.0},sharey=True)
        a=0
        for i, j in combinations(range(num_cols), 2):
            if i != j:
                sns.scatterplot(x=data[:,j], y=data[:,i], alpha=0.6, color="k", ax=axs[a])
                sns.kdeplot(x=data[:,j], y=data[:,i], fill=True, alpha=0.9, ax=axs[a])
                a+=1
                if x_lim is not None:
                    axs[a].set_xlim(x_lim)
                if y_lim is not None:
                    axs[a].set_ylim(y_lim)
        plt.subplots_adjust(wspace=0.05)
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(x=data[:,0], y=data[:,1], alpha=0.6, color="k", ax = ax)
        sns.kdeplot(x=data[:,0], y=data[:,1], fill=True, alpha=0.9, ax = ax)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

    return fig

def plot_kl_divergence_scatter(data,kl_divergence):
    #TODO: This also needs to work for higher dimensional data e.g. 3d
    if torch.is_tensor(data):
        data = data.detach().numpy()
    if torch.is_tensor(kl_divergence):
        kl_divergence = kl_divergence.detach().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    # palette from here:
    #https: // seaborn.pydata.org / tutorial / color_palettes.html
    sns.scatterplot(x=data[:,0], y=data[:,1], hue=kl_divergence, ax = ax, palette='icefire')

    # Create a scalar mappable to show the legend
    # https://stackoverflow.com/questions/62884183/trying-to-add-a-colorbar-to-a-seaborn-scatterplot
    max_deviance = max(abs(kl_divergence)) # ensures that the colorbar is centered
    norm = plt.Normalize(-max_deviance, max_deviance)
    sm = plt.cm.ScalarMappable(cmap="icefire", norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(sm)

    return fig


def plot_splines(layer):

    num_splines = layer.params.size()[1]
    #num_variables = layer.number_variables
    poly_min = layer.polynomial_range[0,0]
    poly_max = layer.polynomial_range[1,0]

    data_span = torch.linspace(poly_min,poly_max,100)
    data_span_vec = data_span.reshape((100,1)).repeat(1,num_splines)

    if layer.type == "transformation":
        output, log_d = layer.forward(data_span_vec, return_log_d=True)
        output_derivativ = torch.exp(log_d).detach().numpy()
    elif layer.type == "decorrelation":
        output = data_span_vec.clone()
        for spline_num in range(num_splines):
            output[:,spline_num] = bspline_prediction(layer.params[:, spline_num],
                               data_span_vec[:, spline_num],
                               degree=layer.degree,
                               polynomial_range=layer.polynomial_range[:, 0], #assume same polly range across variables
                               monotonically_increasing=False,
                               derivativ=0)

    data_span_vec = data_span_vec.detach().numpy()
    output = output.detach().numpy()

    if num_splines > 1 :
        fig, axs = plt.subplots(nrows=1, ncols=num_splines, figsize=(15,5),
                                gridspec_kw={'wspace':0.0, 'hspace':0.0},sharey=True)
        a=0
        for spline_num in range(num_splines):
            sns.lineplot(x=data_span_vec[:,spline_num], y=output[:,spline_num], ax = axs[a])
            if layer.type == "transformation":
                sns.lineplot(x=data_span_vec[:,spline_num], y=output_derivativ[:,spline_num], ax = axs[a])
            axs[a].set_ylim(output.min(), output.max())
            axs[a].set_xlim(poly_min, poly_max)
            a+=1

        plt.subplots_adjust(wspace=0.05)

    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.lineplot(x=data_span_vec.reshape(100), y=output.reshape(100), ax = ax)
        ax.set_ylim(output.min(), output.max())
        ax.set_xlim(poly_min, poly_max)


    return fig