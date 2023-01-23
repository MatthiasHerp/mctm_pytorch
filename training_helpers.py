import torch
from torch import nn
import numpy as np
from torch.distributions import Normal, Laplace
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import seaborn as sns

def objective(y, model, penalty_params, avg = True):
    z, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global = model(y, inverse=False)
    log_likelihood_latent = Normal(0, 1).log_prob(z) # log p_source(z)
    #print(log_likelihood_latent.size())
    #print(log_d.size())
    #print(log_d)

    pen_value_ridge = penalty_params[0]
    pen_first_ridge = penalty_params[1]
    pen_second_ridge = penalty_params[2]

    if avg:
        loss = 1 / (z.size(0)*z.size(1)) * (- log_likelihood_latent - log_d).sum() + \
               pen_second_ridge * second_order_ridge_pen_global + \
               pen_first_ridge * first_order_ridge_pen_global +  \
               pen_value_ridge * param_ridge_pen_global
    else:
        loss = - log_likelihood_latent.sum() - log_d.sum()
    return loss

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def early_stop(self, current_loss):
        if current_loss < self.min_loss:
            self.min_loss = current_loss
            self.counter = 0

        elif np.allclose(current_loss,self.min_loss,self.min_delta):
            self.counter += 1

            if self.counter >= self.patience:
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

def optimize(y, model, objective, penalty_params, iterations = 2000, verbose=False, patience=5, min_delta=1e-7):
    opt = torch.optim.LBFGS(model.parameters(), history_size=1) # no history basically, now the model trains stable, seems simple fischer scoring is enough

    def closure():
        opt.zero_grad()
        neg_log_likelihood = objective(y, model, penalty_params)# use the `objective` function
        neg_log_likelihood.backward() # backpropagate the loss
        return neg_log_likelihood

    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    neg_log_likelihoods = []
    for _ in tqdm(range(iterations)):
        neg_log_likelihood = objective(y, model, penalty_params)
        opt.step(closure)
        neg_log_likelihoods.append(neg_log_likelihood.detach().numpy())

        if verbose:
            print(neg_log_likelihood.item())

        if early_stopper.early_stop(neg_log_likelihood.detach().numpy()):
            print("Early Stop!")
            break

    return neg_log_likelihoods

def train(model, train_data, penalty_params=torch.FloatTensor([0,0,0]), iterations=2000, verbose=True, patience=5, min_delta=1e-7):

    neg_log_likelihoods = optimize(train_data, model, objective, penalty_params = penalty_params, iterations = iterations, verbose=verbose, patience=patience, min_delta=min_delta) # Run training

    # Plot neg_log_likelihoods over training iterations:
    with sns.axes_style('ticks'):
        plt.plot(neg_log_likelihoods)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
    sns.despine(trim = True)


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






