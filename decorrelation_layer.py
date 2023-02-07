import torch
from torch import nn
import numpy as np
from bernstein_transformation_layer import bernstein_prediction
from bspline_prediction import bspline_prediction

def multivariable_lambda_prediction(input, degree, number_variables, params, polynomial_range, spline, inverse=False, span_factor=0.1, span_restriction="None"):

    #steps
    output = input.clone()
    # loop over all variables
    params_index = 0
    # pred penality terms
    second_order_ridge_pen_sum = 0
    first_order_ridge_pen_sum = 0
    param_ridge_pen_sum = 0

    for var_num in range(number_variables):
        #print(var_num)
        # loop over all before variables
        for covar_num in range(var_num):
            #print(covar_num)
            #print(params_index)

            # compute lambda fct value using before variable
            if inverse:
                #output into spline
                if spline == "bspline":
                        lambda_value = bspline_prediction(params[:, params_index],
                                                          output[:, covar_num],
                                                          degree,
                                                          polynomial_range[:, covar_num],
                                                          monotonically_increasing=False,
                                                          derivativ=0,
                                                          span_factor=span_factor,
                                                          span_restriction=span_restriction)

                elif spline == "bernstein":
                    lambda_value = bernstein_prediction(params[:, params_index],
                                                        output[:,covar_num],
                                                        degree,
                                                        polynomial_range[:,covar_num],
                                                        monotonically_increasing=False,
                                                        derivativ=0,
                                                        span_factor=span_factor)
                                                        #return_penalties not implemented yet
            else:
                #input into spline
                if spline == "bspline":
                    lambda_value, second_order_ridge_pen_current, \
                    first_order_ridge_pen_current, param_ridge_pen_current = bspline_prediction(params[:, params_index],
                                                      input[:,covar_num],
                                                      degree,
                                                      polynomial_range[:,covar_num],
                                                      monotonically_increasing=False,
                                                      derivativ=0,
                                                      return_penalties=True,
                                                      span_factor=span_factor,
                                                      span_restriction=span_restriction)
                    second_order_ridge_pen_sum += second_order_ridge_pen_current
                    first_order_ridge_pen_sum += first_order_ridge_pen_current
                    param_ridge_pen_sum += param_ridge_pen_current

                elif spline == "bernstein":
                    lambda_value, second_order_ridge_pen_current, \
                    first_order_ridge_pen_current, param_ridge_pen_current = bernstein_prediction(params[:, params_index],
                                                        input[:,covar_num],
                                                        degree,
                                                        polynomial_range[:,covar_num],
                                                        monotonically_increasing=False,
                                                        derivativ=0,
                                                        span_factor=span_factor)
                    second_order_ridge_pen_sum += second_order_ridge_pen_current
                    first_order_ridge_pen_sum += first_order_ridge_pen_current
                    param_ridge_pen_sum += param_ridge_pen_current

            # update
            # Cloning issue?
            if inverse:
                output[:,var_num] = output[:,var_num] - lambda_value * output[:,covar_num]
            else:
                output[:,var_num] = output[:,var_num] + lambda_value * input[:,covar_num]

            params_index += 1

    if inverse:
        return output
    else:
        return output, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum


class Decorrelation(nn.Module):
    def __init__(self, degree, number_variables, polynomial_range, spline="bspline", span_factor=0.1, span_restriction="None"):
        super().__init__()
        self.type = "decorrelation"
        self.degree  = degree
        self.number_variables = number_variables
        self.polynomial_range = polynomial_range
        self.num_lambdas = number_variables * (number_variables-1) / 2
        self.spline = spline

        self.span_factor = span_factor
        self.span_restriction = span_restriction

        # https://discuss.pytorch.org/t/how-to-turn-list-of-varying-length-tensor-into-a-tensor/1361
        # param dims: 0: basis, 1: variable
        p = torch.FloatTensor(np.repeat(np.repeat(0.1,self.degree+1), self.num_lambdas)) #(torch.rand(int(self.degree+1), int(self.num_lambdas))-0.5)

        if self.num_lambdas == 1:
            self.params = nn.Parameter(p.unsqueeze(1))
        else:
            self.params = nn.Parameter(torch.reshape(p,(self.degree+1, int(self.num_lambdas))))
        #self.params = p

    def forward(self, input, log_d = 0, inverse = False, return_log_d = False, return_penalties=True):

        #if torch.any(torch.abs(input) >= torch.abs(self.polynomial_range)[0]):
        #    print("Warning: input outside of polynomial range")

        if not inverse:
            output, second_order_ridge_pen_sum, \
            first_order_ridge_pen_sum, param_ridge_pen_sum = multivariable_lambda_prediction(input,
                                                     self.degree,
                                                     self.number_variables,
                                                     self.params,
                                                     self.polynomial_range,
                                                     inverse=False,
                                                     spline=self.spline,
                                                     span_factor=self.span_factor,
                                                     span_restriction=self.span_restriction)
        else:
            output = multivariable_lambda_prediction(input,
                                                     self.degree,
                                                     self.number_variables,
                                                     self.params,
                                                     self.polynomial_range,
                                                     inverse=True,
                                                     spline=self.spline,
                                                     span_factor=self.span_factor,
                                                     span_restriction=self.span_restriction)

        if return_log_d and return_penalties:
            return output, log_d, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum
        elif return_log_d:
            return output, log_d
        else:
            return output

    #def __repr__(self):
    #    return "Affine(alpha={alpha:.2f}, beta={beta:.2f})".format(alpha = self.alpha[0], beta = self.beta[0])