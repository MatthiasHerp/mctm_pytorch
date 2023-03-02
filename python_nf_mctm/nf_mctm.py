import torch
from torch import nn
from python_nf_mctm.flip import Flip
from python_nf_mctm.bernstein_transformation_layer import *
from python_nf_mctm.decorrelation_layer import Decorrelation


#class NF_MCTM(nn.Module):
#    def __init__(self, input_min, input_max, polynomial_range, number_variables, spline_decorrelation="bernstein",
#                 degree_transformations=10, degree_decorrelation=12, span_factor=0.1, span_restriction="None",
#                 number_covariates=False): #normalisation_layer=None
#        super(NF_MCTM, self).__init__()
#        self.polynomial_range = polynomial_range
#        self.number_variables = number_variables
#        self.input_min = input_min
#        self.input_max = input_max
#        #self.normalisation_layer = normalisation_layer
#
#        self.degree_transformations = degree_transformations
#        self.degree_decorrelation = degree_decorrelation
#
#        self.span_factor = span_factor
#        self.span_restriction = span_restriction
#
#        # Repeat polynomial ranges for all variables as this is the range for the bsplines essentially
#        polynomial_range_transformation = polynomial_range.repeat(1,self.number_variables)
#        polynomial_range_decorrelation = polynomial_range.repeat(1,self.number_variables)
#
#        self.number_covariates = number_covariates
#
#        #if self.normalisation_layer == "bounding":
#        #    self.l0 = Normalisation(input_min=self.input_min, input_max=self.input_max, output_range=polynomial_range[1]-polynomial_range[1]*0.25)
#        #if self.normalisation_layer == "standardisation":
#        #    self.l0 = Normalisation(input_mean=self.input_min, input_variance=self.input_max, output_range=polynomial_range[1])
#
#        self.l1 = Transformation(degree=self.degree_transformations, number_variables=self.number_variables, polynomial_range=polynomial_range_transformation, span_factor=self.span_factor,
#                                 number_covariates=self.number_covariates)
#        #self.l12 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
#        self.l2 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables, polynomial_range=polynomial_range_decorrelation, span_factor=self.span_factor, span_restriction=self.span_restriction, spline=spline_decorrelation,
#                                number_covariates=self.number_covariates)
#        self.l3 = Flip()
#        #self.l34 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
#        self.l4 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables, polynomial_range=polynomial_range_decorrelation, span_factor=self.span_factor, span_restriction=self.span_restriction, spline=spline_decorrelation,
#                                number_covariates=self.number_covariates)
#        self.l5 = Flip()
#        #self.l56 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
#        self.l6 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables, polynomial_range=polynomial_range_decorrelation, span_factor=self.span_factor, span_restriction=self.span_restriction, spline=spline_decorrelation,
#                                number_covariates=self.number_covariates)
#
#
#    def forward(self, y, covariate=False, train=True, evaluate=True):
#        # Normalisation
#        #if  self.normalisation_layer is not None:
#        #    output = self.l0.forward(y)
#        #else:
#        #    output = y
#
#        # Training or evaluation
#        if train or evaluate:
#
#            if train:
#                # new input false to not recompute basis each iteration
#                output, log_d = self.l1(y, covariate, return_log_d = True, new_input = False)
#            elif evaluate:
#                # new input true as we need to recompute the basis for the validation/test set
#                output, log_d = self.l1(y, covariate, return_log_d=True, new_input = True)
#
#            #output = self.l12(output)
#
#            output, log_d, second_order_ridge_pen_global, \
#            first_order_ridge_pen_global, param_ridge_pen_global, \
#            lambda_matrix_global = self.l2(output, covariate, log_d, return_log_d = True, return_penalties=True)
#
#            output = self.l3(output)
#            #output = self.l34(output)
#
#            output, log_d, second_order_ridge_pen_sum, \
#            first_order_ridge_pen_sum, param_ridge_pen_sum, lambda_matrix = self.l4(output, covariate, log_d, return_log_d = True, return_penalties=True)
#
#            second_order_ridge_pen_global += second_order_ridge_pen_sum
#            first_order_ridge_pen_global += first_order_ridge_pen_sum
#            param_ridge_pen_global += param_ridge_pen_sum
#
#            lambda_matrix_global = torch.matmul(lambda_matrix_global, self.l3(lambda_matrix))
#
#            output = self.l5(output)
#            # output = self.l56(output)
#
#            output, log_d, second_order_ridge_pen_sum, \
#            first_order_ridge_pen_sum, param_ridge_pen_sum, lambda_matrix = self.l6(output, covariate, log_d, return_log_d=True, return_penalties=True)
#
#            second_order_ridge_pen_global += second_order_ridge_pen_sum
#            first_order_ridge_pen_global += first_order_ridge_pen_sum
#            param_ridge_pen_global += param_ridge_pen_sum
#
#            lambda_matrix_global = torch.matmul(lambda_matrix_global, lambda_matrix)
#
#            return output, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global, lambda_matrix_global
#
#        else:
#            # new input true as we need to recompute the basis for the validation/test set
#            output = self.l1(y, covariate, new_input=True)
#            #output = self.l12(output)
#            output = self.l2(output, covariate, return_log_d=False, return_penalties=False)
#            output = self.l3(output)
#            #output = self.l34(output)
#            output = self.l4(output, covariate, return_log_d=False, return_penalties=False)
#            output = self.l5(output)
#            #output = self.l56(output)
#            output = self.l6(output, covariate, return_log_d=False, return_penalties=False)
#
#            return output
#
#    def latent_space_representation(self, y, covariate=False):
#        z = self.forward(y, covariate, train=False, evaluate=False)
#        return z
#
#    def log_likelihood(self, y, covariate=False):
#        #TODO: run this log_likelihood code and the sample code with torch.no_grad() to speed up the code
#        # https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad
#        with torch.no_grad():
#            z, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global, lambda_matrix_global = self.forward(y, covariate=covariate, evaluate=True, train=False)
#            log_likelihood_latent = torch.distributions.Normal(0, 1).log_prob(z)  # log p_source(z)
#            log_likelihood = log_likelihood_latent + log_d #now a minus here
#            vec_log_likelihood = log_likelihood.sum(1)
#        return vec_log_likelihood
#
#    def compute_precision_matrix(self, y, covariate=False):
#
#        with torch.no_grad():
#            z, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, \
#            param_ridge_pen_global, lambda_matrix_global = self.forward(y, covariate=covariate, evaluate=True, train=False)
#
#            # Compute the precision matrix
#            precision_matrix = torch.matmul(torch.transpose(lambda_matrix_global, 1, 2), lambda_matrix_global)
#
#        return precision_matrix
#
#
#    def sample(self, n_samples, covariate=False):
#        z = torch.distributions.Normal(0, 1).sample((n_samples, self.number_variables))
#
#        output = self.l6(z, covariate=covariate, return_log_d=False, return_penalties=False, inverse=True)
#        output = self.l5(output)
#        output = self.l4(output, covariate=covariate, return_log_d=False, return_penalties=False, inverse=True)
#        output = self.l3(output)
#        output = self.l2(output, covariate=covariate, return_log_d=False, return_penalties=False, inverse=True)
#        y = self.l1(output, covariate=covariate, inverse=True)
#
#        return y

class NF_MCTM(nn.Module):
    def __init__(self, input_min, input_max, polynomial_range, number_variables, spline_decorrelation="bernstein",
                 degree_transformations=10, degree_decorrelation=12, span_factor=0.1, span_restriction="None",
                 number_covariates=False, num_decorr_layers=3, list_comprehension=False): #normalisation_layer=None
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

        self.list_comprehension = list_comprehension

        #if self.normalisation_layer == "bounding":
        #    self.l0 = Normalisation(input_min=self.input_min, input_max=self.input_max, output_range=polynomial_range[1]-polynomial_range[1]*0.25)
        #if self.normalisation_layer == "standardisation":
        #    self.l0 = Normalisation(input_mean=self.input_min, input_variance=self.input_max, output_range=polynomial_range[1])

        self.transformation = Transformation(degree=self.degree_transformations, number_variables=self.number_variables,
                                 polynomial_range=polynomial_range_transformation, span_factor=self.span_factor,
                                 number_covariates=self.number_covariates)
        ##self.l12 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
        #self.l2 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables,
        #                        polynomial_range=polynomial_range_decorrelation, span_factor=self.span_factor,
        #                        span_restriction=self.span_restriction, spline=spline_decorrelation,
        #                        number_covariates=self.number_covariates)
        #self.l3 = Flip()
        ##self.l34 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
        #self.l4 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables,
        #                        polynomial_range=polynomial_range_decorrelation, span_factor=self.span_factor,
        #                        span_restriction=self.span_restriction, spline=spline_decorrelation,
        #                        number_covariates=self.number_covariates)
        #self.l5 = Flip()
        ##self.l56 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
        #self.l6 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables,
        #                        polynomial_range=polynomial_range_decorrelation, span_factor=self.span_factor,
        #                        span_restriction=self.span_restriction, spline=spline_decorrelation,
        #                        number_covariates=self.number_covariates)
#
        #self.l7 = Flip()
        ## self.l56 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
        #self.l8 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables,
        #                        polynomial_range=polynomial_range_decorrelation, span_factor=self.span_factor,
        #                        span_restriction=self.span_restriction, spline=spline_decorrelation,
        #                        number_covariates=self.number_covariates)

        self.flip = Flip()

        self.number_decorrelation_layers = num_decorr_layers
        self.decorrelation_layers = nn.ModuleList([Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables,
                                polynomial_range=polynomial_range_decorrelation, span_factor=self.span_factor,
                                span_restriction=self.span_restriction, spline=spline_decorrelation,
                                number_covariates=self.number_covariates,
                                list_comprehension = self.list_comprehension) for i in range(self.number_decorrelation_layers)])

    def forward(self, y, covariate=False, train=True, evaluate=True):
        # Normalisation
        #if  self.normalisation_layer is not None:
        #    output = self.l0.forward(y)
        #else:
        #    output = y

        # Training or evaluation
        if train or evaluate:

            if train:
                # new input false to not recompute basis each iteration
                output, log_d = self.transformation(y, covariate, return_log_d = True, new_input = False)
            elif evaluate:
                # new input true as we need to recompute the basis for the validation/test set
                output, log_d = self.transformation(y, covariate, return_log_d=True, new_input = True)

            #output = self.l12(output)

            lambda_matrix_global = torch.eye(self.number_variables)
            second_order_ridge_pen_global = 0
            first_order_ridge_pen_global = 0
            param_ridge_pen_global = 0

            for i in range(self.number_decorrelation_layers):

                output, log_d, second_order_ridge_pen_sum, \
                first_order_ridge_pen_sum, param_ridge_pen_sum, \
                lambda_matrix = self.decorrelation_layers[i](output, covariate, log_d,
                                                                    return_log_d = True, return_penalties=True)

                second_order_ridge_pen_global += second_order_ridge_pen_sum
                first_order_ridge_pen_global += first_order_ridge_pen_sum
                param_ridge_pen_global += param_ridge_pen_sum

                if (i % 2) == 0:
                    # even
                    lambda_matrix_global = torch.matmul(lambda_matrix_global, self.flip(lambda_matrix))
                else:
                    # odd
                    lambda_matrix_global = torch.matmul(lambda_matrix_global, lambda_matrix)

                output = self.flip(output)

            return output, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global, lambda_matrix_global

        else:
            # new input true as we need to recompute the basis for the validation/test set
            output = self.transformation(y, covariate, new_input=True)

            for i in range(self.number_decorrelation_layers):

                output = self.decorrelation_layers[i](output, covariate)

                output = self.flip(output)

            return output

    def latent_space_representation(self, y, covariate=False):
        z = self.forward(y, covariate, train=False, evaluate=False)
        return z

    def log_likelihood(self, y, covariate=False):
        #TODO: run this log_likelihood code and the sample code with torch.no_grad() to speed up the code
        # https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad
        with torch.no_grad():
            z, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global, lambda_matrix_global = self.forward(y, covariate=covariate, evaluate=True, train=False)
            log_likelihood_latent = torch.distributions.Normal(0, 1).log_prob(z)  # log p_source(z)
            log_likelihood = log_likelihood_latent + log_d #now a minus here
            vec_log_likelihood = log_likelihood.sum(1)
        return vec_log_likelihood

    def compute_precision_matrix(self, y, covariate=False):

        with torch.no_grad():
            z, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, \
            param_ridge_pen_global, lambda_matrix_global = self.forward(y, covariate=covariate, evaluate=True, train=False)

            # Compute the precision matrix
            precision_matrix = torch.matmul(torch.transpose(lambda_matrix_global, 1, 2), lambda_matrix_global)

        return precision_matrix


    def sample(self, n_samples, covariate=False):
        z = torch.distributions.Normal(0, 1).sample((n_samples, self.number_variables))

        for i in range(self.number_decorrelation_layers - 1, -1, -1):
            z = self.flip(z)
            z = self.decorrelation_layers[i](z, covariate=covariate, return_log_d=False, return_penalties=False, inverse=True)

        y = self.transformation(z, covariate=covariate, inverse=True)

        return y