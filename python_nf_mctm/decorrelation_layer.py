import torch
from torch import nn
import numpy as np
from python_nf_mctm.bernstein_transformation_layer import bernstein_prediction
from python_nf_mctm.bspline_prediction import bspline_prediction

def compute_starting_values_bspline(degree,num_lambdas):
    p = torch.FloatTensor(np.repeat(np.repeat(0.1, degree + 1),
                                    num_lambdas))  # (torch.rand(int(self.degree+1), int(self.num_lambdas))-0.5)

    if num_lambdas == 1:
        params = nn.Parameter(p.unsqueeze(1))
    else:
        params = nn.Parameter(torch.reshape(p, (degree + 1, int(num_lambdas))))

    #params = (torch.rand((int(degree + 1), int(num_lambdas))) - 0.5) / 10

    return params

def multivariable_lambda_prediction(input, degree, number_variables, params, polynomial_range, spline, inverse=False, span_factor=0.1, span_restriction="None",
                                    covariate=False,params_covariate=False, list_comprehension=False, dev=False):

    #steps
    output = input.clone()
    # loop over all variables
    params_index = 0
    # pred penality terms
    second_order_ridge_pen_sum = 0
    first_order_ridge_pen_sum = 0
    param_ridge_pen_sum = 0

    # Matrix dimensions:
    # axis 0: number of samples
    # axis 1, 2: number of variables e.g. the lambda matrix of each particular sample
    lambda_matrix_a = torch.eye(number_variables).expand(output.size()[0],number_variables,number_variables)

    if dev is not False:
        lambda_matrix_a.to(dev)

    lambda_matrix = lambda_matrix_a.clone()

    # Inverse needs to be nested for loop as we need it to happen iteratively

    if inverse:

        for var_num in range(number_variables):
            #print(var_num)
            # loop over all before variables
            for covar_num in range(var_num):
                #print(covar_num)
                #print(params_index)

                # compute lambda fct value using before variable
                #output into spline
                if spline == "bspline":
                        #Note: params_index get +=1 at the end so that we always have the right parameters (this needs to be changed for apply vectorisation)
                        lambda_value = bspline_prediction(params[:, params_index],
                                                          output[:, covar_num],
                                                          degree,
                                                          polynomial_range[:, covar_num],
                                                          monotonically_increasing=False,
                                                          derivativ=0,
                                                          return_penalties=False,
                                                          span_factor=span_factor,
                                                          span_restriction=span_restriction,
                                                          covariate=covariate,
                                                          params_covariate=params_covariate[:,covar_num],
                                                          dev=dev)

                elif spline == "bernstein":
                    lambda_value = bernstein_prediction(params[:, params_index],
                                                        output[:,covar_num],
                                                        degree,
                                                        polynomial_range[:,covar_num],
                                                        monotonically_increasing=False,
                                                        derivativ=0,
                                                        return_penalties=False,
                                                        span_factor=span_factor,
                                                        covariate=covariate,
                                                        params_covariate=params_covariate[:,covar_num])
                                                        #return_penalties not implemented yet
                # update
                output[:, var_num] = output[:, var_num] - lambda_value * output[:, covar_num]

                params_index += 1

                # filling the lambda matrix with the computed entries
                lambda_matrix[:, var_num, covar_num] = lambda_value

    # Forward pass e.g. we can vectorize
    else:
        # This is a list comprehension implementation of the foward pass that should be faster than the nested for loop
        if list_comprehension == True:
            def forward_pass_row(var_num, covar_num):

                num_splines = max(var_num * (var_num - 1) / 2,0)
                params_index = int(num_splines + covar_num)

                #input into spline
                if spline == "bspline":
                    lambda_value, second_order_ridge_pen_current, \
                    first_order_ridge_pen_current, param_ridge_pen_current = bspline_prediction(params[:, params_index], #TODO:need index here!
                                                      input[:,covar_num],
                                                      degree,
                                                      polynomial_range[:,covar_num],
                                                      monotonically_increasing=False,
                                                      derivativ=0,
                                                      return_penalties=True,
                                                      span_factor=span_factor,
                                                      span_restriction=span_restriction,
                                                      covariate=covariate,
                                                      params_covariate=params_covariate[:,covar_num],
                                                      dev=dev)
                    #second_order_ridge_pen_sum += second_order_ridge_pen_current
                    #first_order_ridge_pen_sum += first_order_ridge_pen_current
                    #param_ridge_pen_sum += param_ridge_pen_current

                elif spline == "bernstein":
                    lambda_value, second_order_ridge_pen_current, \
                    first_order_ridge_pen_current, param_ridge_pen_current = bernstein_prediction(params[:, params_index],
                                                        input[:,covar_num],
                                                        degree,
                                                        polynomial_range[:,covar_num],
                                                        monotonically_increasing=False,
                                                        derivativ=0,
                                                        return_penalties=True,
                                                        span_factor=span_factor,
                                                        covariate=covariate,
                                                        params_covariate=params_covariate[:,covar_num])
                    #second_order_ridge_pen_sum += second_order_ridge_pen_current
                    #first_order_ridge_pen_sum += first_order_ridge_pen_current
                    #param_ridge_pen_sum += param_ridge_pen_current

                    #lambda_matrix[:, var_num, covar_num] = lambda_value

                return input[:, covar_num] * lambda_value, \
                       second_order_ridge_pen_current, first_order_ridge_pen_current, param_ridge_pen_current,\
                       lambda_value

            def forward_pass_col(var_num):

                if var_num == 0:
                    return torch.zeros(input.size()[0]), 0, 0, 0
                else:
                    res = [forward_pass_row(var_num, covar_num) for covar_num in range(var_num)]

                    add_to_output = sum(res[covar_num][0] for covar_num in range(var_num))

                    second_order_ridge_pen_row_sum = sum(res[covar_num][1] for covar_num in range(var_num))
                    first_order_ridge_pen_row_sum = sum(res[covar_num][2] for covar_num in range(var_num))
                    param_ridge_pen_row_sum = sum(res[covar_num][3] for covar_num in range(var_num))

                    lambda_matrix_entries = torch.cat([res[covar_num][4].unsqueeze(0) for covar_num in range(var_num)])

                    #lambda_value = sum(res[covar_num][4] for covar_num in range(var_num))

                    return add_to_output, \
                              second_order_ridge_pen_row_sum, first_order_ridge_pen_row_sum, param_ridge_pen_row_sum, \
                           lambda_matrix_entries

            res = [forward_pass_col(var_num) for var_num in range(number_variables)]

            output += torch.vstack([res[var_num][0] for var_num in range(number_variables)]).T

            second_order_ridge_pen_sum = sum(res[var_num][1] for var_num in range(number_variables))
            first_order_ridge_pen_sum = sum(res[var_num][2] for var_num in range(number_variables))
            param_ridge_pen_sum = sum(res[var_num][3] for var_num in range(number_variables))

            for var_num in range(1,number_variables): #1 because the first row has no precision matrix entries
                lambda_matrix[:,var_num,0:var_num] = res[var_num][4].T

            # update
        #output[:, var_num] = output[:, var_num] + lambda_value * input[:, covar_num]

        # filling the lambda matrix with the computed entries
        #lambda_matrix[:,var_num,covar_num] = lambda_value

        #[forward_pass_update(var_num, covar_num) for var_num in range(number_variables) for covar_num in range(var_num)]

        else:
            for var_num in range(number_variables):
                # print(var_num)
                # loop over all before variables
                for covar_num in range(var_num):
                    # print(covar_num)
                    # print(params_index)

                    # little test:
                    #num_splines = max(var_num * (var_num - 1) / 2,0)
                    #print(params_index == int(num_splines + covar_num))

                    # compute lambda fct value using before variable
                    # output into spline
                    if spline == "bspline":
                        lambda_value, second_order_ridge_pen_current, \
                        first_order_ridge_pen_current, param_ridge_pen_current = bspline_prediction(
                            params[:, params_index],
                            input[:, covar_num],
                            degree,
                            polynomial_range[:, covar_num],
                            monotonically_increasing=False,
                            derivativ=0,
                            return_penalties=True,
                            span_factor=span_factor,
                            span_restriction=span_restriction,
                            covariate=covariate,
                            params_covariate=params_covariate[:, covar_num],
                            dev=dev)
                        second_order_ridge_pen_sum += second_order_ridge_pen_current
                        first_order_ridge_pen_sum += first_order_ridge_pen_current
                        param_ridge_pen_sum += param_ridge_pen_current


                    elif spline == "bernstein":
                        lambda_value, second_order_ridge_pen_current, \
                        first_order_ridge_pen_current, param_ridge_pen_current = bernstein_prediction(
                            params[:, params_index],
                            input[:, covar_num],
                            degree,
                            polynomial_range[:, covar_num],
                            monotonically_increasing=False,
                            derivativ=0,
                            span_factor=span_factor,
                            covariate=covariate,
                            params_covariate=params_covariate[:, covar_num])
                        second_order_ridge_pen_sum += second_order_ridge_pen_current
                        first_order_ridge_pen_sum += first_order_ridge_pen_current
                        param_ridge_pen_sum += param_ridge_pen_current

                        # return_penalties not implemented yet
                    # update
                    output[:, var_num] = output[:, var_num] + lambda_value * input[:, covar_num]

                    params_index += 1

                    # filling the lambda matrix with the computed entries
                    lambda_matrix[:, var_num, covar_num] = lambda_value

    if inverse:
        return output
    else:
        return output, \
               second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum,\
               lambda_matrix

class Decorrelation(nn.Module):
    def __init__(self, degree, number_variables, polynomial_range, spline="bspline", span_factor=0.1, span_restriction="None",
                 number_covariates=False, list_comprehension = False,
                 dev=False):
        super().__init__()
        self.type = "decorrelation"
        self.degree  = degree
        self.number_variables = number_variables
        self.polynomial_range = polynomial_range
        self.num_lambdas = number_variables * (number_variables-1) / 2
        self.spline = spline

        self.span_factor = span_factor
        self.span_restriction = span_restriction

        self.dev = dev

        self.params = compute_starting_values_bspline(self.degree, self.num_lambdas)

        self.number_covariates = number_covariates

        self.list_comprehension = list_comprehension

        if self.number_covariates is not False:
            if self.number_covariates > 1:
                print("Warning, covariates not implemented for more than 1 covariate")
            self.params_covariate = compute_starting_values_bspline(self.degree, self.num_lambdas)
        else:
            self.params_covariate = False


    def forward(self, input, covariate=False, log_d = 0, inverse = False, return_log_d = False, return_penalties=True):

        #if torch.any(torch.abs(input) >= torch.abs(self.polynomial_range)[0]):
        #    print("Warning: input outside of polynomial range")

        if not inverse:
            output, second_order_ridge_pen_sum, \
            first_order_ridge_pen_sum, param_ridge_pen_sum, \
            lambda_matrix = multivariable_lambda_prediction(input,
                                                     self.degree,
                                                     self.number_variables,
                                                     self.params,
                                                     self.polynomial_range,
                                                     inverse=False,
                                                     spline=self.spline,
                                                     span_factor=self.span_factor,
                                                     span_restriction=self.span_restriction,
                                                     covariate = covariate,
                                                     params_covariate = self.params_covariate,
                                                     list_comprehension = self.list_comprehension,
                                                     dev = self.dev)
        else:
            output = multivariable_lambda_prediction(input,
                                                     self.degree,
                                                     self.number_variables,
                                                     self.params,
                                                     self.polynomial_range,
                                                     inverse=True,
                                                     spline=self.spline,
                                                     span_factor=self.span_factor,
                                                     span_restriction=self.span_restriction,
                                                     covariate = covariate,
                                                     params_covariate = self.params_covariate,
                                                     list_comprehension=self.list_comprehension,
                                                     dev = self.dev)

        if return_log_d and return_penalties:
            return output, log_d, second_order_ridge_pen_sum, first_order_ridge_pen_sum, param_ridge_pen_sum, lambda_matrix
        elif return_log_d:
            return output, log_d
        else:
            return output

    #def __repr__(self):
    #    return "Affine(alpha={alpha:.2f}, beta={beta:.2f})".format(alpha = self.alpha[0], beta = self.beta[0])