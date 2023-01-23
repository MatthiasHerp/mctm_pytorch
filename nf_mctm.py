import torch
from torch import nn
from flip import Flip
from bernstein_transformation_layer import Transformation
from decorrelation_layer import Decorrelation

class InvertibleSequential(nn.Sequential):
    def forward(self, x, inverse):
        for module in self._modules.values():
            x = module(x, inverse)
        return x

class NF_MCTM(nn.Module):
    def __init__(self, polynomial_range, number_variables, number_transformation_layers=1, num_decorrelation_layers=3, spline_decorrelation="bernstein"):
        super(NF_MCTM, self).__init__()
        self.polynomial_range = polynomial_range
        self.number_variables = number_variables

        self.network = InvertibleSequential()

        for i in range(number_transformation_layers):
            self.network.add_module("transformation_{}".format(i), Transformation(degree=10,
                                                                     number_variables=self.number_variables,
                                                                     polynomial_range=self.polynomial_range.repeat(1,self.number_variables)))

        for i in range(num_decorrelation_layers):
            self.network.add_module("decorrelation_{}".format(i), Decorrelation(degree=12,
                                                                                number_variables=self.number_variables,
                                                                                polynomial_range=self.polynomial_range.repeat(1,self.number_variables),
                                                                                spline=spline_decorrelation))
            # After each decorrelation layer, we flip the variables, except for the last one (doesnt really matter as after thte transformation each variable is iid standard normal)
            if i < num_decorrelation_layers:
                self.network.add_module("flip_{}".format(i), Flip())

    def forward(self, x, inverse=False):
        input_tuple = (x,0,0,0,0)
        return self.network(input_tuple, inverse=inverse)





#class NF_MCTM(nn.Module):
#    def __init__(self, polynomial_range, number_variables, spline_decorrelation="bernstein", calc_method="torch_bspline"):
#        super(NF_MCTM, self).__init__()
#        self.polynomial_range = polynomial_range
#        self.number_variables = number_variables
#
#        self.l1 = Transformation(degree=10, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3))
#        self.l2 = Decorrelation(degree=12, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3), spline=spline_decorrelation, calc_method=calc_method)
#        self.l3 = Flip()
#        self.l4 = Decorrelation(degree=12, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3), spline=spline_decorrelation, calc_method=calc_method)
#        self.l5 = Flip()
#        self.l6 = Decorrelation(degree=12, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3), spline=spline_decorrelation, calc_method=calc_method)
#
#
#    def forward(self, x, train=True):
#        if train:
#            output, log_d = self.l1(x, return_log_d = True)
#
#            output, log_d, second_order_ridge_pen_global, \
#            first_order_ridge_pen_global, param_ridge_pen_global = self.l2(output, log_d, return_log_d = True, return_penalties=True)
#
#            output = self.l3(output)
#
#            output, log_d, second_order_ridge_pen_sum, \
#            first_order_ridge_pen_sum, param_ridge_pen_sum = self.l4(output, log_d, return_log_d = True, return_penalties=True)
#            output = self.l5(output)
#            second_order_ridge_pen_global += second_order_ridge_pen_sum
#            first_order_ridge_pen_global += first_order_ridge_pen_sum
#            param_ridge_pen_global += param_ridge_pen_sum
#
#            output, log_d, second_order_ridge_pen_sum, \
#            first_order_ridge_pen_sum, param_ridge_pen_sum = self.l6(output, log_d, return_log_d=True, return_penalties=True)
#            second_order_ridge_pen_global += second_order_ridge_pen_sum
#            first_order_ridge_pen_global += first_order_ridge_pen_sum
#            param_ridge_pen_global += param_ridge_pen_sum
#
#            return output, log_d, second_order_ridge_pen_global, first_order_ridge_pen_global, param_ridge_pen_global
#
#        else:
#            output = self.l1(x)
#            output = self.l2(output, return_log_d=False, return_penalties=False)
#            output = self.l3(output)
#            output = self.l4(output, return_log_d=False, return_penalties=False)
#            output = self.l5(output)
#            output = self.l6(output, return_log_d=False, return_penalties=False)
#
#            return output