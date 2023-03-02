import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import pandas as pd
import random
import warnings
from scipy import stats
import time
from torch.distributions.multivariate_normal import MultivariateNormal
from pingouin import multivariate_normality
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from python_nf_mctm.bspline_prediction import *

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

def plot_densities(data,covariate=False,x_lim=None,y_lim=None):

    # Ensures that by default all points are in the plot and axis have the same span (not distortion, can see distribution clearly)
    if x_lim is None:
        x_lim = [data.min(),data.max()]
    if y_lim is None:
        y_lim = [data.min(),data.max()]

    num_cols = data.shape[1]
    num_combinations = int(num_cols * (num_cols - 1) / 2)
    if covariate is False:
        numbers_covariates = 1
    else:

        #binning the covariates
        # as explained here: https://discuss.pytorch.org/t/binning-tensor-values/89998
        covariate = torch.round(covariate/0.2)*0.2
        numbers_covariates = torch.unique(covariate).size(0)
        covariate_values = torch.unique(covariate)

    if num_combinations > 1 :
        fig, axs = plt.subplots(nrows=numbers_covariates, ncols=num_combinations, figsize=(15,5),
                                gridspec_kw={'wspace':0.01, 'hspace':0.0})
        a=0
        for i, j in combinations(range(num_cols), 2):
            if i != j:
                if covariate is False:
                    sns.scatterplot(x=data[:,j], y=data[:,i], alpha=0.6, color="k", ax=axs[a])
                    #TODO:  got ValueError: Contour levels must be increasing
                    # suggestion on whre it came from: https://stackoverflow.com/questions/62233779/valueerror-contour-levels-must-be-increasing-contour-plot-in-python
                    #sns.kdeplot(x=data[:, j], y=data[:, i], fill=True, alpha=0.9, ax=axs[a])
                else:
                    for c in range(numbers_covariates):
                        sub_data = data[covariate == covariate_values[c]]
                        sub_covariate = covariate[covariate == covariate_values[c]]
                        sns.scatterplot(x=sub_data[:, j], y=sub_data[:, i], hue=sub_covariate, alpha=0.6, color="k", ax=axs[c,a])
                        sns.kdeplot(x=sub_data[:,j], y=sub_data[:,i], fill=True, alpha=0.9, ax=axs[c,a])

                        axs[c,a].set_xlim(x_lim)
                        axs[c,a].set_ylim(y_lim)

                a += 1

        plt.subplots_adjust(wspace=0.05)
    else:
        fig, ax = plt.subplots(nrows=numbers_covariates, figsize=(6, 6*numbers_covariates))
        if covariate is False:
            sns.scatterplot(x=data[:,0], y=data[:,1], alpha=0.6, color="k", ax=ax)
            sns.kdeplot(x=data[:, 0], y=data[:, 1], fill=True, alpha=0.9, ax=ax)
        else:
            for c in range(numbers_covariates):
                sub_data = data[covariate == covariate_values[c]]
                sub_covariate = covariate[covariate == covariate_values[c]]
                sns.scatterplot(x=sub_data[:, 0], y=sub_data[:, 1], hue=sub_covariate, alpha=0.6, color="k",
                                ax=ax[c])
                sns.kdeplot(x=sub_data[:, 0], y=sub_data[:, 1], fill=True, alpha=0.9, ax=ax[c,])
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

    return fig

#kl_divergence_scatter
def plot_metric_scatter(data, metric, covariate=False, x_lim=None, y_lim=None, metric_type="kl_divergence"):
    # Ensures that by default all points are in the plot and axis have the same span (not distortion, can see distribution clearly)
    if x_lim is None:
        x_lim = [data.min(), data.max()]
    if y_lim is None:
        y_lim = [data.min(), data.max()]

    if torch.is_tensor(data):
        data = data.detach().numpy()
    if torch.is_tensor(metric):
        metric = metric.detach().numpy()

    num_cols = data.shape[1]
    num_combinations = int(num_cols * (num_cols - 1) / 2)

    if num_combinations > 1:
        fig, axs = plt.subplots(nrows=1, ncols=num_combinations, figsize=(15, 5),
                                gridspec_kw={'wspace': 0.01, 'hspace': 0.0})
        a = 0
        for i, j in combinations(range(num_cols), 2):
            if i != j:
                if covariate is not False:
                    warnings.warn("Covariate is not supported for 3d data yet")

                max_deviance = np.max(abs(metric))  # ensures that the colorbar is centered
                norm = plt.Normalize(-max_deviance, max_deviance)

                # palette from here:
                # https: // seaborn.pydata.org / tutorial / color_palettes.html
                if metric_type == "kl_divergence":
                    sns.scatterplot(x=data[:, i], y=data[:, j], hue=metric, ax=axs[a], palette='icefire')
                elif metric_type == "precision_matrix":
                    sns.scatterplot(x=data[:, i], y=data[:, j], hue=metric[:,i,j], hue_norm=norm, ax=axs[a], palette='icefire')

                # Create a scalar mappable to show the legend
                # https://stackoverflow.com/questions/62884183/trying-to-add-a-colorbar-to-a-seaborn-scatterplot
                sm = plt.cm.ScalarMappable(cmap="icefire", norm=norm)
                sm.set_array([])

                axs[a].set_xlim(x_lim)
                axs[a].set_ylim(y_lim)

                axs[a].set_xlabel("y_" + str(i))
                axs[a].set_ylabel("y_" + str(j))

                # Remove the legend and add a colorbar
                axs[a].get_legend().remove()

                a += 1

        axs[num_combinations-1].figure.colorbar(sm)

    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        # palette from here:
        #https: // seaborn.pydata.org / tutorial / color_palettes.html
        if metric_type == "kl_divergence":
            sns.scatterplot(x=data[:,0], y=data[:,1], hue=metric, ax = ax, palette='icefire')
        elif metric_type == "precision_matrix":
            sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=metric[:, 0, 1], ax=ax, palette='icefire')

        # Create a scalar mappable to show the legend
        # https://stackoverflow.com/questions/62884183/trying-to-add-a-colorbar-to-a-seaborn-scatterplot
        max_deviance = np.max(abs(metric)) # ensures that the colorbar is centered
        norm = plt.Normalize(-max_deviance, max_deviance)
        sm = plt.cm.ScalarMappable(cmap="icefire", norm=norm)
        sm.set_array([])

        ax.set_xlabel("y_" + str(0))
        ax.set_ylabel("y_" + str(1))

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        ax.figure.colorbar(sm)

    return fig

def plot_metric_hist(metric, covariate=False):

    if torch.is_tensor(metric):
        metric = metric.detach().numpy()

    num_cols = metric.shape[1]
    num_combinations = int(num_cols * (num_cols - 1) / 2)

    if num_combinations > 1:
        fig, axs = plt.subplots(nrows=1, ncols=num_combinations, figsize=(15, 5),
                                gridspec_kw={'wspace': 0.01, 'hspace': 0.0})
        a = 0
        for i, j in combinations(range(num_cols), 2):
            if i != j:
                if covariate is not False:
                    warnings.warn("Covariate is not supported for 3d data yet")

                # palette from here:
                # https: // seaborn.pydata.org / tutorial / color_palettes.html
                sns.histplot(x=metric[:,i,j], ax=axs[a])

                axs[a].set_xlabel("y_" + str(i))
                axs[a].set_ylabel("y_" + str(j))

                a += 1
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        # palette from here:
        #https: // seaborn.pydata.org / tutorial / color_palettes.html
        sns.histplot(x=metric[:,0,1], ax=ax)

        ax.set_xlabel("y_" + str(0))
        ax.set_ylabel("y_" + str(1))

    return fig


def plot_splines(layer, y_train=None, covariate_exists=False):

    num_splines = layer.params.size()[1]
    #num_variables = layer.number_variables

    if layer.type == "transformation":
        poly_min = y_train.min(0).values
        poly_max = y_train.max(0).values
    elif layer.type == "decorrelation":
        poly_min = layer.polynomial_range[0,:]
        poly_max = layer.polynomial_range[1,:]


    data_span_vec = torch.zeros((1000,num_splines), dtype=torch.float32)
    for i in range(num_splines):
        data_span_vec[:,i] = torch.linspace(poly_min[i],poly_max[i],1000)

    if layer.type == "transformation":
        results = pd.DataFrame(columns=["y", "y_estimated", "z_tilde", "z_tilde_derivativ", "covariate", "spline_num"])
        if covariate_exists is True:
            for cov_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
                covariate_value = torch.Tensor([cov_value]).repeat(100)
                z_tilde, log_d = layer.forward(data_span_vec, covariate=covariate_value, return_log_d=True)
                z_tilde_derivativ = torch.exp(log_d)
                data_span_vec_estimated = layer.forward(z_tilde, covariate=covariate_value,  inverse=True)
                #data_span_vec_estimated = data_span_vec_estimated.detach().numpy()

                for spline_num in range(num_splines):
                    results = results.append(pd.DataFrame({"y": data_span_vec.detach().numpy()[:, spline_num],
                                    "y_estimated": data_span_vec_estimated.detach().numpy()[:, spline_num],
                                    "z_tilde": z_tilde.detach().numpy()[:, spline_num],
                                    "z_tilde_derivativ": z_tilde_derivativ.detach().numpy()[:, spline_num],
                                    "covariate": cov_value, "spline_num": spline_num}), ignore_index=True)

        else:
            results = results.drop(("covariate"), axis=1)
            z_tilde, log_d = layer.forward(data_span_vec, covariate=False, return_log_d=True)
            z_tilde_derivativ = torch.exp(log_d)
            data_span_vec_estimated = layer.forward(z_tilde, covariate=False, inverse=True)
            # data_span_vec_estimated = data_span_vec_estimated.detach().numpy()

            for spline_num in range(num_splines):
                results = results.append(pd.DataFrame({"y": data_span_vec.detach().numpy()[:, spline_num],
                                                       "y_estimated": data_span_vec_estimated.detach().numpy()[:,
                                                                      spline_num],
                                                       "z_tilde": z_tilde.detach().numpy()[:, spline_num],
                                                       "z_tilde_derivativ": z_tilde_derivativ.detach().numpy()[:,spline_num],
                                                       "spline_num": spline_num }),ignore_index=True)

    elif layer.type == "decorrelation":
        z_tilde = data_span_vec.clone()
        results = pd.DataFrame(columns=["y", "y_estimated", "z_tilde", "z_tilde_derivativ", "covariate", "spline_num"])
        if covariate_exists is True:
            for cov_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
                covariate_value = torch.Tensor([cov_value]).repeat(100)
                for spline_num in range(num_splines):
                    z_tilde[:,spline_num] = bspline_prediction(layer.params[:, spline_num],
                                       data_span_vec[:, spline_num],
                                       degree=layer.degree,
                                       polynomial_range=layer.polynomial_range[:, 0], #assume same polly range across variables
                                       monotonically_increasing=False,
                                       derivativ=0,
                                       covariate=covariate_value,
                                       params_covariate=layer.params_covariate[:, 0]) # hardcoded for only one covariate
                    results = results.append(pd.DataFrame({"y": data_span_vec.detach().numpy()[:, spline_num],
                                                           "z_tilde": z_tilde.detach().numpy()[:, spline_num],
                                                           "covariate": cov_value,
                                                           "spline_num": spline_num }), ignore_index=True)
        else:
            results = results.drop(("covariate"), axis=1)

            for spline_num in range(num_splines):
                z_tilde[:, spline_num] = bspline_prediction(layer.params[:, spline_num],
                                                            data_span_vec[:, spline_num],
                                                            degree=layer.degree,
                                                            polynomial_range=layer.polynomial_range[:, 0],
                                                            # assume same polly range across variables
                                                            monotonically_increasing=False,
                                                            derivativ=0,
                                                            covariate=False,
                                                            params_covariate=False)  # hardcoded for only one covariate
                results = results.append(pd.DataFrame({"y": data_span_vec.detach().numpy()[:, spline_num],
                                                       "z_tilde": z_tilde.detach().numpy()[:, spline_num],
                                                       "spline_num": spline_num}), ignore_index=True)


    #data_span_vec = data_span_vec.detach().numpy()
    #z_tilde = z_tilde.detach().numpy()

    if num_splines > 1 :
        fig, axs = plt.subplots(nrows=1, ncols=num_splines, figsize=(15,5),
                                gridspec_kw={'wspace':0.01, 'hspace':0.0},sharey=False, sharex=False)#dont want to share all x (sharex) and y axis (sharey)
        a=0
        for spline_num in range(num_splines):
            subset_results = results[results["spline_num"]==spline_num]
            if covariate_exists is True:
                sns.lineplot(x="y", y="z_tilde", hue="covariate", data=subset_results, ax = axs[a])
                if layer.type == "transformation":
                    sns.lineplot(x="y", y="z_tilde_derivativ", hue="covariate", data=subset_results, ax=axs[a])
                    sns.lineplot(x="y_estimated", y="z_tilde", hue="covariate", linestyle='--', data=subset_results,
                                 ax=axs[a])
            else:
                sns.lineplot(x="y", y="z_tilde", data=subset_results, ax = axs[a])
                if layer.type == "transformation":
                    sns.lineplot(x="y", y="z_tilde_derivativ", data=subset_results, ax=axs[a])
                    sns.lineplot(x="y_estimated", y="z_tilde", linestyle='--', data=subset_results,
                                 ax=axs[a])
            #TODO: solve this issue, somehow the min() max() in the comment below does not span all possible values
            #if layer.type == "transformation":
            #    axs[a].set_ylim(subset_results["z_tilde"].min(), subset_results["z_tilde"].max())
            #elif layer.type == "decorrelation":
            #    axs[a].set_ylim(-2, 2)
            if layer.type == "transformation":
                axs[a].set_ylim(subset_results["z_tilde"].min(), subset_results["z_tilde"].max())
                axs[a].set_xlim(subset_results["y"].min(), subset_results["y"].max())

                axs[a].set_xlabel("y_" + str(spline_num))
                axs[a].set_ylabel("z_tilde" + str(spline_num))

            elif layer.type == "decorrelation":
                axs[a].set_ylim(subset_results["z_tilde"].min(), subset_results["z_tilde"].max())
                axs[a].set_xlim(subset_results["y"].min(), subset_results["y"].max())

                axs[a].set_xlabel("z_tilde")
                axs[a].set_ylabel("lambda_" + str(spline_num))

            a+=1

        plt.subplots_adjust(wspace=0.05)

    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        if covariate_exists is True:
            sns.lineplot(x="y", y="z_tilde", hue="covariate", data=results, ax = ax)
        else:
            sns.lineplot(x="y", y="z_tilde", data=results, ax = ax)
        ax.set_ylim(results["z_tilde"].min(), results["z_tilde"].max())
        ax.set_xlim(results["y"].min(), results["y"].max())

        ax.set_xlabel("z_tilde")
        ax.set_ylabel("lambda_" + str(0))


    return fig