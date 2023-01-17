import torch
from torch import nn
from flip import Flip
from bernstein_transformation_layer import Transformation
from decorrelation_layer import Decorrelation



class NF_MCTM(nn.Module):
    def __init__(self, polynomial_range, number_variables):
        super(NF_MCTM, self).__init__()
        self.polynomial_range = polynomial_range
        self.number_variables = number_variables

        self.l1 = Transformation(degree=10, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3))
        self.l2 = Decorrelation(degree=12, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3), spline="bernstein")
        self.l3 = Flip()
        self.l4 = Decorrelation(degree=12, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3), spline="bernstein")
        self.l5 = Flip()
        self.l6 = Decorrelation(degree=12, number_variables=self.number_variables, polynomial_range=self.polynomial_range.repeat(1,3), spline="bernstein")


    def forward(self, x, return_log_d=False):
        if return_log_d==True:
            output, log_d = self.l1(x, return_log_d = True)
            output, log_d = self.l2(output, log_d, return_log_d = True)
            output = self.l3(output)
            output, log_d = self.l4(output, log_d, return_log_d = True)
            output = self.l5(output)
            output, log_d = self.l6(output, log_d, return_log_d=True)

            return output, log_d
        else:
            output = self.l1(x)
            output = self.l2(output)
            output = self.l3(output)
            output = self.l4(output)
            output = self.l5(output)
            output = self.l6(output)

            return output