import torch
from torch import nn
import numpy as np
from torch.distributions import Normal, Laplace
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import seaborn as sns
from bernstein_transformation_layer import bernstein_prediction
from bspline_prediction import bspline_prediction

def multivariable_lambda_prediction(input, degree, number_variables, params, polynomial_range, inverse=False, spline="bspline"):

    #steps
    output = input.clone()
    # loop over all variables
    params_index = 0

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
                    lambda_value = bspline_prediction(params[:, params_index], output[:,covar_num], degree, polynomial_range[:,covar_num], monotonically_increasing=False, derivativ=0)
                elif spline == "bernstein":
                    lambda_value = bernstein_prediction(params[:, params_index], output[:,covar_num], degree, polynomial_range[:,covar_num], monotonically_increasing=False, derivativ=0)
            else:
                #input into spline
                if spline == "bspline":
                    lambda_value = bspline_prediction(params[:, params_index], input[:,covar_num], degree, polynomial_range[:,covar_num], monotonically_increasing=False, derivativ=0)
                elif spline == "bernstein":
                    lambda_value = bernstein_prediction(params[:, params_index], input[:,covar_num], degree, polynomial_range[:,covar_num], monotonically_increasing=False, derivativ=0)

            # update
            # Cloning issue?
            if inverse:
                output[:,var_num] = output[:,var_num] - lambda_value * output[:,covar_num]
            else:
                output[:,var_num] = output[:,var_num] + lambda_value * input[:,covar_num]

            params_index += 1

    return output

class Decorrelation(nn.Module):
    def __init__(self, degree, number_variables, polynomial_range, spline="bspline"):
        super().__init__()
        self.degree  = degree
        self.number_variables = number_variables
        self.polynomial_range = polynomial_range
        self.num_lambdas = number_variables * (number_variables-1) / 2
        self.spline = spline
        # https://discuss.pytorch.org/t/how-to-turn-list-of-varying-length-tensor-into-a-tensor/1361
        # param dims: 0: basis, 1: variable
        p = torch.FloatTensor(np.repeat(np.repeat(0.1,self.degree+1), self.num_lambdas))

        if self.num_lambdas == 1:
            self.params = nn.Parameter(p.unsqueeze(1))
        else:
            self.params = nn.Parameter(torch.reshape(p,(self.degree+1, int(self.num_lambdas))))

    def forward(self, input, log_d = 0, inverse = False):
        if not inverse:
            output = multivariable_lambda_prediction(input, self.degree, self.number_variables, self.params, self.polynomial_range, inverse=False, spline=self.spline)
        else:
            output = multivariable_lambda_prediction(input, self.degree, self.number_variables, self.params, self.polynomial_range, inverse=True, spline=self.spline)
        return output, log_d

    #def __repr__(self):
    #    return "Affine(alpha={alpha:.2f}, beta={beta:.2f})".format(alpha = self.alpha[0], beta = self.beta[0])