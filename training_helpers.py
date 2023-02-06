import time

import torch
from torch import nn
import numpy as np
from torch.distributions import Normal, Laplace
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import seaborn as sns
from pytorch_lbfgs.LBFGS import LBFGS, FullBatchLBFGS

def objective(y, model, penalty_params, avg = True):
    z, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global = model(y, train=True)
    log_likelihood_latent = Normal(0, 1).log_prob(z) # log p_source(z)
    #print(log_likelihood_latent.size())
    #print(log_d.size())
    #print(log_d)

    pen_value_ridge = penalty_params[0] * param_ridge_pen_global
    pen_first_ridge = penalty_params[1] * first_order_ridge_pen_global
    pen_second_ridge = penalty_params[2] * second_order_ridge_pen_global

    neg_likelihood = (- log_likelihood_latent - log_d).sum()

    if avg:
        # We average the loss and the penalties
        # By averaging the penalties we make the penalisation magnitude independent of the number of knots
        pen_value_ridge = 1 / (3*model.degree_decorrelation) * pen_value_ridge
        pen_first_ridge = 1 / (3*model.degree_decorrelation) * pen_first_ridge
        pen_second_ridge = 1 / (3*model.degree_decorrelation) * pen_second_ridge

        neg_likelihood = 1 / (z.size(0)*z.size(1)) * neg_likelihood

    loss = neg_likelihood + \
           pen_value_ridge + \
           pen_first_ridge +  \
           pen_second_ridge

    return loss, pen_value_ridge, pen_first_ridge, pen_second_ridge

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0., global_min_loss=-np.inf):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf
        self.global_min_loss = global_min_loss

    def early_stop(self, current_loss):
        if current_loss < self.min_loss:
            self.min_loss = current_loss
            self.counter = 0

        elif np.allclose(current_loss,self.min_loss,self.min_delta):
            self.counter += 1

            if self.counter >= self.patience:
                return True

        if current_loss < self.global_min_loss:
            return True

        return False

#def optimize(y, model, objective, iterations = 2000, verbose=True):
#    opt                 = optim.Adam(model.parameters(), lr = 1e-2)
#    scheduler           = optim.lr_scheduler.StepLR(opt, step_size = 500, gamma = 0.8)
#    neg_log_likelihoods = []
#    for _ in tqdm(range(iterations)):
#        opt.zero_grad() # zero out gradients first on the optimizer
#        neg_log_likelihood = objective(y, model) # use the `objective` function
#        neg_log_likelihood.backward() # backpropagate the loss
#        opt.step()
#        scheduler.step()
#        neg_log_likelihoods.append(neg_log_likelihood.detach().numpy())
#        if verbose:
#            print(neg_log_likelihood.item())
#
#    return neg_log_likelihoods

def optimize(y, model, objective, penalty_params, learning_rate=1, iterations = 2000, verbose=False, patience=5, min_delta=1e-7, global_min_loss=0.01):
    opt = FullBatchLBFGS(model.parameters(), lr=1., history_size=1, line_search='Wolfe')
    #opt = torch.optim.LBFGS(model.parameters(), lr=learning_rate, history_size=1) # no history basically, now the model trains stable, seems simple fischer scoring is enough

    def closure():
        opt.zero_grad()
        loss, pen_value_ridge, \
        pen_first_ridge, pen_second_ridge  = objective(y, model, penalty_params) # use the `objective` function
        #loss.backward() # backpropagate the loss
        return loss

    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, global_min_loss=global_min_loss)

    loss = closure()
    loss.backward()
    options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}

    loss_list = []
    for i in tqdm(range(iterations)):
        number_iterations = i

        current_loss, pen_value_ridge, \
        pen_first_ridge, pen_second_ridge = objective(y, model, penalty_params)
        opt.step(options) # Note: if options not included you get the error: if 'damping' not in options.keys(): AttributeError: 'function' object has no attribute 'keys'
        loss_list.append(current_loss.detach().numpy().item())

        if verbose:
            print("Loss:",current_loss.item())

        if early_stopper.early_stop(current_loss.detach().numpy()):
            print("Early Stop at iteration", i, "with loss", current_loss.item(), "and patience", patience, "and min_delta", min_delta)
            break

    return loss_list, number_iterations, pen_value_ridge, pen_first_ridge, pen_second_ridge

def train(model, train_data, penalty_params=torch.FloatTensor([0,0,0]), learning_rate=1, iterations=2000, verbose=True, patience=5, min_delta=1e-7, return_report=True):

    if return_report:
        start = time.time()
        loss_list, number_iterations, pen_value_ridge, pen_first_ridge, pen_second_ridge = optimize(train_data, model, objective, penalty_params = penalty_params, learning_rate=learning_rate, iterations = iterations, verbose=verbose, patience=patience, min_delta=min_delta) # Run training
        end = time.time()

        training_time = end - start

        # Plot neg_log_likelihoods over training iterations:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.lineplot(data=loss_list, ax=ax)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        return loss_list, number_iterations, pen_value_ridge, pen_first_ridge, pen_second_ridge, training_time, fig

    else:
        loss_list, number_iterations, pen_value_ridge, pen_first_ridge, pen_second_ridge = optimize(train_data, model, objective, penalty_params = penalty_params, learning_rate=learning_rate, iterations = iterations, verbose=verbose, patience=patience, min_delta=min_delta) # Run training


#TODO: Outdated function when we merely tested with Laplace example from probML lecture
def evaluate(model):
    p_source = Normal(0, 1)
    p_target = Laplace(5, 3)
    x_true   = p_target.sample((2000, 2)) # samples to compare to

    # Generate samples from source distribution
    z = p_source.sample((2000, 2))

    # Use our trained model get samples from the target distribution
    x_flow, log_d = model.forward(z, inverse=True)

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






