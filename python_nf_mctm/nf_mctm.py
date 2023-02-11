import torch
from torch import nn
from python_nf_mctm.flip import Flip
from python_nf_mctm.bernstein_transformation_layer import *
from python_nf_mctm.decorrelation_layer import Decorrelation


class NF_MCTM(nn.Module):
    def __init__(self, input_min, input_max, polynomial_range, number_variables, spline_decorrelation="bernstein",
                 degree_transformations=10, degree_decorrelation=12, span_factor=0.1, span_restriction="None",
                 number_covariates=False): #normalisation_layer=None
        super(NF_MCTM, self).__init__()
        self.polynomial_range = polynomial_range
        self.number_variables = number_variables
        self.input_min = input_min
        self.input_max = input_max
        #self.normalisation_layer = normalisation_layer

        self.degree_transformations = degree_transformations
        self.degree_decorrelation = degree_decorrelation

        self.span_factor = span_factor
        self.span_restriction = span_restriction

        # Repeat polynomial ranges for all variables as this is the range for the bsplines essentially
        polynomial_range_transformation = polynomial_range.repeat(1,self.number_variables)
        polynomial_range_decorrelation = polynomial_range.repeat(1,self.number_variables)

        self.number_covariates = number_covariates

        #if self.normalisation_layer == "bounding":
        #    self.l0 = Normalisation(input_min=self.input_min, input_max=self.input_max, output_range=polynomial_range[1]-polynomial_range[1]*0.25)
        #if self.normalisation_layer == "standardisation":
        #    self.l0 = Normalisation(input_mean=self.input_min, input_variance=self.input_max, output_range=polynomial_range[1])

        self.l1 = Transformation(degree=self.degree_transformations, number_variables=self.number_variables, polynomial_range=polynomial_range_transformation, span_factor=self.span_factor,
                                 number_covariates=self.number_covariates)
        #self.l12 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
        self.l2 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables, polynomial_range=polynomial_range_decorrelation, span_factor=self.span_factor, span_restriction=self.span_restriction, spline=spline_decorrelation,
                                number_covariates=self.number_covariates)
        self.l3 = Flip()
        #self.l34 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
        self.l4 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables, polynomial_range=polynomial_range_decorrelation, span_factor=self.span_factor, span_restriction=self.span_restriction, spline=spline_decorrelation,
                                number_covariates=self.number_covariates)
        self.l5 = Flip()
        #self.l56 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
        self.l6 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables, polynomial_range=polynomial_range_decorrelation, span_factor=self.span_factor, span_restriction=self.span_restriction, spline=spline_decorrelation,
                                number_covariates=self.number_covariates)


    def forward(self, y, covariate=False, train=True):
        # Normalisation
        #if  self.normalisation_layer is not None:
        #    output = self.l0.forward(y)
        #else:
        #    output = y

        # Training or evaluation
        if train:
            output, log_d = self.l1(y, covariate, return_log_d = True)

            #output = self.l12(output)

            output, log_d, second_order_ridge_pen_global, \
            first_order_ridge_pen_global, param_ridge_pen_global = self.l2(output, covariate, log_d, return_log_d = True, return_penalties=True)

            output = self.l3(output)
            #output = self.l34(output)

            output, log_d, second_order_ridge_pen_sum, \
            first_order_ridge_pen_sum, param_ridge_pen_sum = self.l4(output, covariate, log_d, return_log_d = True, return_penalties=True)

            output = self.l5(output)
            #output = self.l56(output)
            second_order_ridge_pen_global += second_order_ridge_pen_sum
            first_order_ridge_pen_global += first_order_ridge_pen_sum
            param_ridge_pen_global += param_ridge_pen_sum

            output, log_d, second_order_ridge_pen_sum, \
            first_order_ridge_pen_sum, param_ridge_pen_sum = self.l6(output, covariate, log_d, return_log_d=True, return_penalties=True)
            second_order_ridge_pen_global += second_order_ridge_pen_sum
            first_order_ridge_pen_global += first_order_ridge_pen_sum
            param_ridge_pen_global += param_ridge_pen_sum

            return output, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global

        else:
            output = self.l1(y, covariate)
            #output = self.l12(output)
            output = self.l2(output, covariate, return_log_d=False, return_penalties=False)
            output = self.l3(output)
            #output = self.l34(output)
            output = self.l4(output, covariate, return_log_d=False, return_penalties=False)
            output = self.l5(output)
            #output = self.l56(output)
            output = self.l6(output, covariate, return_log_d=False, return_penalties=False)

            return output

    def latent_space_representation(self, y, covariate=False):
        z = self.forward(y, covariate, train=False)
        return z

    def log_likelihood(self, y):
        #TODO: run this log_likelihood code and the sample code with torch.no_grad() to speed up the code
        # https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad
        # with torch.no_grad():
        z, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global = self.forward(y,train=True)
        log_likelihood_latent = torch.distributions.Normal(0, 1).log_prob(z)  # log p_source(z)
        log_likelihood = log_likelihood_latent + log_d #now a minus here
        vec_log_likelihood = log_likelihood.sum(1)
        return vec_log_likelihood

    def sample(self, n_samples, covariate=False):
        z = torch.distributions.Normal(0, 1).sample((n_samples, self.number_variables))

        output = self.l6(z, covariate=covariate, return_log_d=False, return_penalties=False, inverse=True)
        output = self.l5(output)
        output = self.l4(output, covariate=covariate, return_log_d=False, return_penalties=False, inverse=True)
        output = self.l3(output)
        output = self.l2(output, covariate=covariate, return_log_d=False, return_penalties=False, inverse=True)
        y = self.l1(output, covariate=covariate, inverse=True)

        return y
