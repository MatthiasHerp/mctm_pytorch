import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import seaborn as sns

from python_nf_mctm.training_helpers import EarlyStopper
from python_nf_mctm.bspline_prediction import bspline_prediction
from python_nf_mctm.bernstein_prediction import bernstein_prediction
from pytorch_lbfgs.LBFGS import LBFGS, FullBatchLBFGS
from python_nf_mctm.splines_utils import adjust_ploynomial_range

def multivariable_bernstein_prediction(input, degree, number_variables, params, polynomial_range, monotonically_increasing, spline, derivativ=0, span_factor=0.1,
                                       covariate=None,params_covariate=None):
    # input dims: 0: observation number, 1: variable
    # cloning tipp from here: https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/10
    output = input.clone()

    second_order_ridge_pen_sum = 0
    first_order_ridge_pen_sum = 0
    param_ridge_pen_sum = 0

    for var_num in range(number_variables):

        if spline == "bernstein":
            output[:,var_num], second_order_ridge_pen_current, \
            first_order_ridge_pen_current, param_ridge_pen_current = bernstein_prediction(params[:,var_num], input[:,var_num], degree, polynomial_range[:,var_num], monotonically_increasing, derivativ, span_factor=span_factor,
                                                                                          covariate=covariate, params_covariate=params_covariate)
        elif spline == "bspline":
            output[:,var_num], second_order_ridge_pen_current, \
            first_order_ridge_pen_current, param_ridge_pen_current = bspline_prediction(params[:,var_num], input[:,var_num], degree, polynomial_range[:,var_num], monotonically_increasing, derivativ, return_penalties=True, span_factor=span_factor,
                                                                                        covariate=covariate, params_covariate=params_covariate)

        second_order_ridge_pen_sum += second_order_ridge_pen_current
        first_order_ridge_pen_sum += first_order_ridge_pen_current
        param_ridge_pen_sum += param_ridge_pen_current

    return output, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum

def compute_starting_values_berstein_polynomials(degree,min,max,number_variables):
    par_restricted_opt = torch.linspace(min,max,degree+1)
    par_unristricted = par_restricted_opt
    par_unristricted[1:] = torch.log(par_restricted_opt[1:] - par_restricted_opt[:-1])#torch.diff(par_restricted_opt[1:]))

    par_restricted_opt = torch.Tensor.repeat(par_unristricted,(number_variables,1)).T
    #par_restricted_opt = torch.reshape(par_restricted_opt,(degree+1,3))

    return par_restricted_opt

class Transformation(nn.Module):
    def __init__(self, degree, number_variables, polynomial_range, monotonically_increasing=True, spline="bernstein", span_factor=0.1,
                 covariate=None, params_covariate=None):
        super().__init__()
        self.type = "transformation"
        self.degree  = degree
        self.number_variables = number_variables
        self.polynomial_range = polynomial_range
        self.spline = spline
        # param dims: 0: basis, 1: variable
        self.params = nn.Parameter(compute_starting_values_berstein_polynomials(degree,
                                                                                polynomial_range[0,0],
                                                                                polynomial_range[1,0],
                                                                                self.number_variables))
        self.monotonically_increasing = monotonically_increasing

        self.span_factor = span_factor

        self.covariate = covariate
        self.params_covariate = params_covariate

    def forward(self, input, log_d = 0, inverse = False, return_log_d = False):
        # input dims: 0: observaton number, 1: variable
        if not inverse:
            output, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum = multivariable_bernstein_prediction(input, self.degree, self.number_variables, self.params, self.polynomial_range, self.monotonically_increasing, spline=self.spline, span_factor=self.span_factor)
            output_first_derivativ, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen = multivariable_bernstein_prediction(input, self.degree, self.number_variables, self.params, self.polynomial_range, self.monotonically_increasing, spline=self.spline, derivativ=1, span_factor=self.span_factor)
            log_d = log_d + torch.log(output_first_derivativ) # Error this is false we require the derivativ of the bernstein polynomial!332'
            # took out torch.abs(), misunderstanding, determinant can be a negativ value (flipping of the coordinate system)
        else:
            output, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum = multivariable_bernstein_prediction(input, self.degree_inverse, self.number_variables, self.params_inverse, self.polynomial_range_inverse, self.monotonically_increasing_inverse, spline=self.spline_inverse, span_factor=self.span_factor,
                                                                                                                                    covariate=self.covariate, params_covariate=self.params_covariate)

        if return_log_d==True:
            return output, log_d
        else:
            return output

    def approximate_inverse(self, input, monotonically_increasing_inverse=True, spline_inverse="bernstein", degree_inverse=0, iterations=1000, lr=1, patience=20, min_delta=1e-4, global_min_loss=0.001, span_factor_inverse=0.1):
        # optimization using linespace data and the forward berstein polynomial?

        if degree_inverse == 0:
            degree_inverse = 2 * self.degree

        self.monotonically_increasing_inverse = monotonically_increasing_inverse
        self.span_factor_inverse = span_factor_inverse

        #a, b = torch.meshgrid([torch.linspace(input[:,0].min(),input[:,0].max(),100),torch.linspace(input[:,1].min(),input[:,1].max(),100)])
        #input_space = torch.vstack([a.flatten(),b.flatten()]).T

        input_space = torch.vstack([torch.linspace(input[:,0].min(),input[:,0].max(),10000),torch.linspace(input[:,1].min(),input[:,1].max(),10000)]).T

        output_space = self.forward(input_space)

        span_0 = output_space[:, 0].max() - output_space[:, 0].min()
        span_1 = output_space[:, 1].max() - output_space[:, 1].min()
        polynomial_range_inverse = torch.tensor([[output_space[:, 0].min() - span_0*span_factor_inverse, output_space[:, 1].min() - span_1*span_factor_inverse],
                                                 [output_space[:, 0].max() + span_0*span_factor_inverse, output_space[:, 1].max() + span_1*span_factor_inverse]], dtype=torch.float32)

        #input_space = input
        #output_space = multivariable_bernstein_prediction(input_space, self.degree, self.number_variables, self.params, monotonically_increasing=True)

        inv_trans = Transformation(degree=degree_inverse,
                                   number_variables=self.number_variables,
                                   polynomial_range=polynomial_range_inverse,
                                   monotonically_increasing=monotonically_increasing_inverse,
                                   spline=spline_inverse)

        #def se(y_estimated, y_train):
        #    return torch.sum((y_train - y_estimated)**2)

        #loss_mse = se()
        opt = FullBatchLBFGS(inv_trans.parameters(), lr=lr, history_size=1, line_search="Wolfe")

        loss_fct = nn.L1Loss()  # MSELoss L1Loss
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, global_min_loss=global_min_loss)


        def closure():
            opt.zero_grad()
            input_space_pred = inv_trans.forward(output_space.detach())
            loss = loss_fct(input_space_pred, input_space.detach())  # use the `objective` function
            #loss.backward(retain_graph=True)  # backpropagate the loss
            return loss

        loss = closure()
        loss.backward()
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}

        #opt_inv  = optim.Adam(inv_trans.parameters(), lr = lr, weight_decay=weight_decay)
        #scheduler_inv = optim.lr_scheduler.StepLR(opt_inv, step_size = 500, gamma = 0.5)
        loss_list = []

        for i in tqdm(range(iterations)):

            # needs to be computed manually at each step
            #input_space_comp = input_space
            #output_space, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum = multivariable_bernstein_prediction(input_space_comp, self.degree, self.number_variables, self.params, self.polynomial_range, monotonically_increasing=True)

            #opt_inv.zero_grad() # zero out gradients first on the optimizer
            #input_space_pred = inv_trans.forward(output_space.detach())
            #current_loss = loss_fct(input_space_pred, input_space.detach())
            #l2_losses.append(current_loss.detach().numpy())

            #current_loss = loss_mse(input_space_pred, input_space_comp) # use the `objective` function

            #current_loss.backward() # backpropagate the loss
            #opt_inv.step()
            #scheduler_inv.step()

            #opt.step(closure)

            current_loss, _, _, _, _, _, _, _ = opt.step(options)
            loss_list.append(current_loss.detach().numpy().item())

            if early_stopper.early_stop(current_loss=current_loss.detach().numpy(), model=inv_trans):
                print("Early Stop at iteration", i, "with loss", current_loss.item(), "and patience", patience,
                      "and min_delta", min_delta)
                break

        # Return the best model which is not necessarily the last model
        inv_trans = Transformation(degree=degree_inverse,
                                   number_variables=self.number_variables,
                                   polynomial_range=polynomial_range_inverse,
                                   monotonically_increasing=monotonically_increasing_inverse,
                                   spline=spline_inverse)

        inv_trans.load_state_dict(early_stopper.best_model_state)

        #input_space_pred_final = inv_trans.forward(output_space.detach())
        #loss_final = loss_fct(input_space_pred_final, input_space.detach())


        print("Final loss", early_stopper.min_loss)

        # Plot neg_log_likelihoods over training iterations:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.lineplot(data=loss_list, ax=ax)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        self.polynomial_range_inverse = polynomial_range_inverse
        self.params_inverse = inv_trans.params
        self.spline_inverse = spline_inverse
        self.degree_inverse = degree_inverse

        return fig

    #TODO: repr needs to be redone
    def __repr__(self):
        return "Transformation(degree={degree:.2f}, params={params:.2f})".format(degree = self.degree, params = self.params)