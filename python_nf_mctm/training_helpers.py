import time
import copy
import torch
import warnings
from torch import nn
import numpy as np
from torch.distributions import Normal, Laplace
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import seaborn as sns
from pytorch_lbfgs.LBFGS import LBFGS, FullBatchLBFGS
import copy

def objective(y, model, penalty_params, lambda_penalty_params: torch.Tensor =False, train_covariates=False, avg = True):
    #TODO: take the outputted lambda matrix and penalize it based on an additional lasso precision matrix pen matrix
    #      needs option to pass a symmetric matrix (with diag of 1s) of penalizations or one value applied to alll non dieagonal elements
    z, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global, lambda_matrix_global = model(y, covariate=train_covariates, train=True)
    log_likelihood_latent = Normal(0, 1).log_prob(z) # log p_source(z)
    #print(log_likelihood_latent.size())
    #print(log_d.size())
    #print(log_d)

    pen_value_ridge = penalty_params[0] * param_ridge_pen_global
    pen_first_ridge = penalty_params[1] * first_order_ridge_pen_global
    pen_second_ridge = penalty_params[2] * second_order_ridge_pen_global

    # Penalty for the Lambda Matrix,
    # note that for the other penalties the ridge is already computed in layers here we first need to do absolute values
    if lambda_penalty_params is not False:
        if torch.diag(lambda_penalty_params, 0).sum() != 0:
            warnings.warn("Warning: diagonal of lambda penalty matrix is not zero")
        if not (lambda_penalty_params.transpose(0, 1) == lambda_penalty_params).all():
            warnings.warn("Warning: lambda penalty matrix is not symmetric")
        # Note: need compute precision matrix from lambda matrix and mean here was we did not do that in the layers
        precision_matrix = torch.matmul(torch.transpose(lambda_matrix_global, 1, 2), lambda_matrix_global)
        pen_lambda_lasso = (lambda_penalty_params.unsqueeze(0) * torch.abs(precision_matrix)).mean()
    else:
        pen_lambda_lasso = 0

    neg_likelihood = (- log_likelihood_latent - log_d).sum()

    if avg:
        # need to compute number of parameters to average correctly
        number_variables = y.size(1)
        number_lambda_matrix_entries = number_variables * (number_variables - 1) / 2
        number_params_lambda = 3 * number_lambda_matrix_entries * model.degree_decorrelation
        # For decorrelatioon layer degree == number of parameters

        # We average the loss and the penalties
        # By averaging the penalties we make the penalisation magnitude independent of the number of knots
        pen_value_ridge = pen_value_ridge / number_params_lambda
        pen_first_ridge = pen_first_ridge / number_params_lambda
        pen_second_ridge = pen_second_ridge / number_params_lambda

        pen_lambda_lasso = pen_lambda_lasso / number_lambda_matrix_entries

        neg_likelihood = 1 / (z.size(0)*z.size(1)) * neg_likelihood

    loss = neg_likelihood + \
           pen_value_ridge + \
           pen_first_ridge +  \
           pen_second_ridge + \
           pen_lambda_lasso

    return loss, pen_value_ridge, pen_first_ridge, pen_second_ridge, pen_lambda_lasso

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0., global_min_loss=-np.inf):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf
        self.global_min_loss = global_min_loss

    def early_stop(self, current_loss, model):
        #print(self.counter)
        #if current_loss < self.min_loss:
            #print("current loss:",current_loss," smaller than min_loss:",self.min_loss)
            #self.best_model_state = copy.deepcopy(model.state_dict())
            #self.min_loss = current_loss
            #self.counter = 0
        print(self.counter)

        if current_loss < self.min_loss - self.min_delta:
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.min_loss = current_loss
            self.counter = 0

        else:
            self.counter += 1

            if self.counter >= self.patience:
                print("Early stopping due to no improvement in loss for",self.patience,"iterations")
                return True

        if current_loss < self.global_min_loss:
            print("Early stopping due to global minimum loss reached")
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

def optimize(y_train, y_validate, model, objective, penalty_params, lambda_penalty_params=False, train_covariates=False, validate_covariates=False, learning_rate=1, iterations = 2000, verbose=False, patience=5, min_delta=1e-7, global_min_loss=-np.inf, optimizer='LBFGS'):
    opt = FullBatchLBFGS(model.parameters(), lr=learning_rate, history_size=1, line_search='Wolfe')
    #opt = torch.optim.LBFGS(model.parameters(), lr=learning_rate, history_size=1) # no history basically, now the model trains stable, seems simple fischer scoring is enough

    def closure():
        opt.zero_grad()
        loss, pen_value_ridge, \
        pen_first_ridge, pen_second_ridge, \
        pen_lambda_lasso  = objective(y_train, model, penalty_params, lambda_penalty_params=lambda_penalty_params, train_covariates=train_covariates) # use the `objective` function
        #loss.backward() # backpropagate the loss
        return loss

    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, global_min_loss=global_min_loss)

    loss = closure()
    loss.backward()
    options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}

    if optimizer == "Adam":
        opt = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size = 500, gamma = 0.8)

    loss_list = []
    model_val = copy.deepcopy(model)
    for i in tqdm(range(iterations)):
        number_iterations = i

        if optimizer == "Adam":
            opt.zero_grad()
            loss, pen_value_ridge, \
            pen_first_ridge, pen_second_ridge, \
            pen_lambda_lasso = objective(y_train, model, penalty_params, lambda_penalty_params=lambda_penalty_params, train_covariates=train_covariates)
            loss.backward()
            opt.step()
            scheduler.step()
            current_loss = loss
        elif optimizer == "LBFGS":
            current_loss, _, _, _, _, _, _, _ = opt.step(options) # Note: if options not included you get the error: if 'damping' not in options.keys(): AttributeError: 'function' object has no attribute 'keys'
        loss_list.append(current_loss.detach().numpy().item())

        #model_val.state_dict() = copy.deepcopy(model.state_dict())
        #if i > 3000 and i % 10 == 0:
        #    model_val.load_state_dict(model.state_dict())
        #    current_neg_log_likl_validation = -1*torch.mean(model_val.log_likelihood(y_validate, covariate=validate_covariates))
        #    current_loss = current_neg_log_likl_validation
        #    print(current_neg_log_likl_validation)
#
        #    if early_stopper.early_stop(current_neg_log_likl_validation.detach().numpy(), model):  # current_loss
        #        print("Early Stop at iteration", i, "with loss", current_loss.item(), "and patience", patience,
        #              "and min_delta", min_delta)
        #        break

        if early_stopper.early_stop(current_loss.detach().numpy(), model):  # current_loss
            print("Early Stop at iteration", i, "with loss", current_loss.item(), "and patience", patience,
                  "and min_delta", min_delta)
            break

        if verbose:
            print("Loss:",current_loss.item())


    # Return the best model which is not necessarily the last model
    model.load_state_dict(early_stopper.best_model_state)

    # Rerun model at the end to get final penalties
    _, pen_value_ridge, pen_first_ridge, pen_second_ridge, pen_lambda_lasso = objective(y_train, model, penalty_params, lambda_penalty_params=lambda_penalty_params, train_covariates=train_covariates)

    return loss_list, number_iterations, pen_value_ridge, pen_first_ridge, pen_second_ridge, pen_lambda_lasso

def train(model, train_data, validate_data, train_covariates=False, validate_covariates=False, penalty_params=torch.FloatTensor([0,0,0]), lambda_penalty_params=False, learning_rate=1, iterations=2000, verbose=True, patience=5, min_delta=1e-7, return_report=True,
          optimizer='LBFGS'):

    if return_report:
        start = time.time()
        loss_list, number_iterations, \
        pen_value_ridge, pen_first_ridge, pen_second_ridge, pen_lambda_lasso = optimize(train_data, validate_data, model, objective, train_covariates=train_covariates, validate_covariates=validate_covariates, penalty_params = penalty_params, lambda_penalty_params=lambda_penalty_params,
                                                                                        learning_rate=learning_rate, iterations = iterations, verbose=verbose, patience=patience, min_delta=min_delta, optimizer=optimizer) # Run training
        end = time.time()

        training_time = end - start

        # Plot neg_log_likelihoods over training iterations:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.lineplot(data=loss_list, ax=ax)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        return loss_list, number_iterations, pen_value_ridge, pen_first_ridge, pen_second_ridge, pen_lambda_lasso, training_time, fig

    else:
        loss_list, number_iterations, pen_value_ridge, pen_first_ridge, pen_second_ridge, pen_lambda_lasso = optimize(train_data, validate_data, model, objective, train_covariates=train_covariates, penalty_params = penalty_params, lambda_penalty_params=lambda_penalty_params,
                                                                                                                      learning_rate=learning_rate, iterations = iterations, verbose=verbose, patience=patience, min_delta=min_delta) # Run training


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






