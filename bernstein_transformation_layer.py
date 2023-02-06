import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import seaborn as sns

from training_helpers import EarlyStopper
from bspline_prediction import bspline_prediction
from bernstein_prediction import bernstein_prediction
from pytorch_lbfgs.LBFGS import LBFGS, FullBatchLBFGS

def multivariable_bernstein_prediction(input, degree, number_variables, params, polynomial_range, monotonically_increasing, spline, derivativ=0, span_factor=0.1):
    # input dims: 0: observation number, 1: variable
    # cloning tipp from here: https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/10
    output = input.clone()

    second_order_ridge_pen_sum = 0
    first_order_ridge_pen_sum = 0
    param_ridge_pen_sum = 0

    for var_num in range(number_variables):

        if spline == "bernstein":
            output[:,var_num], second_order_ridge_pen_current, \
            first_order_ridge_pen_current, param_ridge_pen_current = bernstein_prediction(params[:,var_num], input[:,var_num], degree, polynomial_range[:,var_num], monotonically_increasing, derivativ, span_factor=span_factor)
        elif spline == "bspline":
            output[:,var_num], second_order_ridge_pen_current, \
            first_order_ridge_pen_current, param_ridge_pen_current = bspline_prediction(params[:,var_num], input[:,var_num], degree, polynomial_range[:,var_num], monotonically_increasing, derivativ, return_penalties=True, span_factor=span_factor)

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
    def __init__(self, degree, number_variables, polynomial_range, monotonically_increasing=True, spline="bernstein", span_factor=0.1):
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

    def forward(self, input, log_d = 0, inverse = False, return_log_d = False):
        # input dims: 0: observaton number, 1: variable
        if not inverse:
            output, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum = multivariable_bernstein_prediction(input, self.degree, self.number_variables, self.params, self.polynomial_range, self.monotonically_increasing, spline=self.spline, span_factor=self.span_factor)
            output_first_derivativ, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen = multivariable_bernstein_prediction(input, self.degree, self.number_variables, self.params, self.polynomial_range, self.monotonically_increasing, spline=self.spline, derivativ=1, span_factor=self.span_factor)
            log_d = log_d + torch.log(output_first_derivativ) # Error this is false we require the derivativ of the bernstein polynomial!332'
            # took out torch.abs(), misunderstanding, determinant can be a negativ value (flipping of the coordinate system)
        else:
            output, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum = multivariable_bernstein_prediction(input, self.degree_inverse, self.number_variables, self.params_inverse, self.polynomial_range_inverse, self.monotonically_increasing_inverse, spline=self.spline_inverse, span_factor=self.span_factor)

        if return_log_d==True:
            return output, log_d
        else:
            return output

    def approximate_inverse(self, input, monotonically_increasing_inverse=True, spline_inverse="bernstein", degree_inverse="None", iterations=1000, patience=5, min_delta=1e-4, global_min_loss=0.001):
        # optimization using linespace data and the forward berstein polynomial?

        if degree_inverse == "None":
            degree_inverse = 2 * self.degree

        self.monotonically_increasing_inverse = monotonically_increasing_inverse

        #a, b = torch.meshgrid([torch.linspace(input[:,0].min(),input[:,0].max(),100),torch.linspace(input[:,1].min(),input[:,1].max(),100)])
        #input_space = torch.vstack([a.flatten(),b.flatten()]).T

        input_space = torch.vstack([torch.linspace(input[:,0].min(),input[:,0].max(),10000),torch.linspace(input[:,1].min(),input[:,1].max(),10000)]).T

        output_space = self.forward(input_space)

        polynomial_range_inverse = torch.tensor([[output_space[:, 0].min()-2, output_space[:, 1].min()-2],
                                                 [output_space[:, 0].max()+2, output_space[:, 1].max()+2]], dtype=torch.float32)

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
        opt = FullBatchLBFGS(inv_trans.parameters(), lr=1., history_size=1, line_search="Wolfe")

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
        l2_losses = []

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
            l2_losses.append(current_loss.detach().numpy())

            if early_stopper.early_stop(current_loss.detach().numpy()):
                print("Early Stop at iteration", i, "with loss", current_loss.item(), "and patience", patience,
                      "and min_delta", min_delta)
                break

        print("Final loss", current_loss.item())

        with sns.axes_style('ticks'):
            plt.plot(l2_losses)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
        sns.despine(trim = True)

        self.polynomial_range_inverse = polynomial_range_inverse
        self.params_inverse = inv_trans.params
        self.spline_inverse = spline_inverse
        self.degree_inverse = degree_inverse

    #TODO: repr needs to be redone
    def __repr__(self):
        return "Transformation(degree={degree:.2f}, params={params:.2f})".format(degree = self.degree, params = self.params)