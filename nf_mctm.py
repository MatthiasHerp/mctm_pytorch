import torch
from torch import nn
from flip import Flip
from bernstein_transformation_layer import Transformation
from decorrelation_layer import Decorrelation
from reluler_layer import ReLULeR
from normalisation import Normalisation



class NF_MCTM(nn.Module):
    def __init__(self, input_min, input_max, polynomial_range, number_variables, spline_decorrelation="bernstein",
                 degree_transformations=10, degree_decorrelation=12): #normalisation_layer=None
        #TODO: define polynomial ranges for the transformation and the decorrelation layers separetly
        super(NF_MCTM, self).__init__()
        self.polynomial_range = polynomial_range
        self.number_variables = number_variables
        self.input_min = input_min
        self.input_max = input_max
        #self.normalisation_layer = normalisation_layer

        self.degree_transformations = degree_transformations
        self.degree_decorrelation = degree_decorrelation

        #if self.normalisation_layer == "bounding":
        #    self.l0 = Normalisation(input_min=self.input_min, input_max=self.input_max, output_range=polynomial_range[1]-polynomial_range[1]*0.25)
        #if self.normalisation_layer == "standardisation":
        #    self.l0 = Normalisation(input_mean=self.input_min, input_variance=self.input_max, output_range=polynomial_range[1])

        self.l1 = Transformation(degree=self.degree_transformations, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3))
        #self.l12 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
        self.l2 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3), spline=spline_decorrelation)
        self.l3 = Flip()
        #self.l34 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
        self.l4 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3), spline=spline_decorrelation)
        self.l5 = Flip()
        #self.l56 = ReLULeR(polynomial_range_abs=self.polynomial_range[1])
        self.l6 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3), spline=spline_decorrelation)


    def forward(self, y, train=True):
        # Normalisation
        #if  self.normalisation_layer is not None:
        #    output = self.l0.forward(y)
        #else:
        #    output = y

        # Training or evaluation
        if train:
            output, log_d = self.l1(y, return_log_d = True)

            #output = self.l12(output)

            output, log_d, second_order_ridge_pen_global, \
            first_order_ridge_pen_global, param_ridge_pen_global = self.l2(output, log_d, return_log_d = True, return_penalties=True)

            output = self.l3(output)
            #output = self.l34(output)

            output, log_d, second_order_ridge_pen_sum, \
            first_order_ridge_pen_sum, param_ridge_pen_sum = self.l4(output, log_d, return_log_d = True, return_penalties=True)

            output = self.l5(output)
            #output = self.l56(output)
            second_order_ridge_pen_global += second_order_ridge_pen_sum
            first_order_ridge_pen_global += first_order_ridge_pen_sum
            param_ridge_pen_global += param_ridge_pen_sum

            output, log_d, second_order_ridge_pen_sum, \
            first_order_ridge_pen_sum, param_ridge_pen_sum = self.l6(output, log_d, return_log_d=True, return_penalties=True)
            second_order_ridge_pen_global += second_order_ridge_pen_sum
            first_order_ridge_pen_global += first_order_ridge_pen_sum
            param_ridge_pen_global += param_ridge_pen_sum

            return output, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global

        else:
            output = self.l1(y)
            #output = self.l12(output)
            output = self.l2(output, return_log_d=False, return_penalties=False)
            output = self.l3(output)
            #output = self.l34(output)
            output = self.l4(output, return_log_d=False, return_penalties=False)
            output = self.l5(output)
            #output = self.l56(output)
            output = self.l6(output, return_log_d=False, return_penalties=False)

            return output

    def latent_space_representation(self, y):
        z = self.forward(y,train=False)
        return z

    def log_likelihood(self, y):
        z, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global = self.forward(y,train=True)
        log_likelihood_latent = torch.distributions.Normal(0, 1).log_prob(z)  # log p_source(z)
        log_likelihood = log_likelihood_latent + log_d #now a minus here
        vec_log_likelihood = log_likelihood.sum(1)
        return vec_log_likelihood

    def sample(self, n_samples):
        z = torch.distributions.Normal(0, 1).sample((n_samples, self.number_variables))

        output = self.l6(z, return_log_d=False, return_penalties=False)
        output = self.l5(output)
        output = self.l4(output, return_log_d=False, return_penalties=False)
        output = self.l3(output)
        output = self.l2(output, return_log_d=False, return_penalties=False)
        y = self.l1(output)

        return y


