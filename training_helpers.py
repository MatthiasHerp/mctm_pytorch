import torch
from torch import nn
import numpy as np
from torch.distributions import Normal, Laplace
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import seaborn as sns

def objective(y, mctm, avg = True):
    z, log_d = mctm(y)
    log_likelihood_latent = Normal(0, 1).log_prob(z) # log p_source(z)
    #print(log_likelihood_latent.size())
    #print(log_d.size())
    #print(log_d)
    if avg:
        loss = 1 / (z.size(0)*z.size(1)) * (- log_likelihood_latent - log_d).sum()
    else:
        loss = - log_likelihood_latent.sum() - log_d.sum()
    return loss

def optimize(y, mctm, objective, iterations = 2000, verbose=True):
    opt                 = optim.Adam(mctm.parameters(), lr = 1e-2)
    scheduler           = optim.lr_scheduler.StepLR(opt, step_size = 500, gamma = 0.8)
    neg_log_likelihoods = []
    for _ in tqdm(range(iterations)):
        opt.zero_grad() # zero out gradients first on the optimizer
        neg_log_likelihood = objective(y, mctm) # use the `objective` function
        neg_log_likelihood.backward() # backpropagate the loss
        opt.step()
        scheduler.step()
        neg_log_likelihoods.append(neg_log_likelihood.detach().numpy())
        if verbose:
            print(neg_log_likelihood.item())

    return neg_log_likelihoods

#def optimize(y, mctm, objective, iterations = 2000):
#    opt = torch.optim.LBFGS(mctm.parameters())
#
#    def closure():
#        opt.zero_grad()
#        neg_log_likelihood = objective(y, mctm) # use the `objective` function
#        neg_log_likelihood.backward() # backpropagate the loss
#        return neg_log_likelihood
#
#    neg_log_likelihoods = []
#    for _ in tqdm(range(iterations)):
#        neg_log_likelihood = objective(y, mctm)
#        opt.step(closure)
#        neg_log_likelihoods.append(neg_log_likelihood.detach().numpy())
#    return neg_log_likelihoods

def train(mctm, train_data, iterations=2000, verbose=True):

    neg_log_likelihoods = optimize(train_data, mctm, objective, iterations = iterations, verbose=verbose) # Run training

    # Plot neg_log_likelihoods over training iterations:
    with sns.axes_style('ticks'):
        plt.plot(neg_log_likelihoods)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
    sns.despine(trim = True)


def evaluate(mctm):
    p_source = Normal(0, 1)
    p_target = Laplace(5, 3)
    x_true   = p_target.sample((2000, 2)) # samples to compare to

    # Generate samples from source distribution
    z = p_source.sample((2000, 2))

    # Use our trained model get samples from the target distribution
    x_flow, log_d = mctm.forward(z, inverse=True)

    # Plot histogram of training samples `x` and generated samples `x_flow` to compare the two.
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize = (4,4), dpi = 150)
        ax.hist(x_true[:,0].detach().numpy().ravel(), bins = 50, alpha = 0.5,
                histtype = 'step', label = "true", density = True);
        ax.hist(x_flow[:,0].detach().numpy().ravel(), bins = 50, alpha = 0.5,
                histtype = 'step', label = "flow", density = True);
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend(loc = "upper right")
    sns.despine(trim = True)






