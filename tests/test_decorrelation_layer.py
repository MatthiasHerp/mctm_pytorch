import matplotlib.pyplot as plt
import unittest

from python_nf_mctm.training_helpers import *
from python_nf_mctm.decorrelation_layer import *
from python_nf_mctm.simulation_study_helpers import *


class Decorrelation_Model(nn.Module):
    def __init__(self, polynomial_range, number_variables, degree_transformations=10, degree_decorrelation=10, span_factor=0.1, span_restriction=None, spline_decorrlation="bspline"):
        super(Decorrelation_Model, self).__init__()
        self.polynomial_range = polynomial_range
        self.number_variables = number_variables

        self.degree_transformations = degree_transformations
        self.degree_decorrelation = degree_decorrelation

        self.spline_decorrelation = spline_decorrlation
        self.span_factor = span_factor
        self.span_restriction = span_restriction

        self.l2 = Decorrelation(degree=self.degree_decorrelation, number_variables=self.number_variables, polynomial_range=polynomial_range, span_factor=self.span_factor, span_restriction=self.span_restriction, spline=self.spline_decorrelation)


    def forward(self, y, train=True):
        if train:
            output, log_d = self.l2(y, return_log_d=True, return_penalties=False)
            return output, log_d, 0, 0, 0
        else:
            output = self.l2(y, return_log_d=False, return_penalties=False)
            return output


class TestFit(unittest.TestCase):

    def test_1(self):
        """
        span_restriction:"None"
        """

        loc = torch.zeros(2)
        lam = torch.Tensor([[1, 0],
                            [3, 1]])
        scale = lam @ torch.eye(2) @ torch.transpose(lam, 0, 1)
        y_distribution = MultivariateNormal(loc, scale)
        y = y_distribution.sample((2000,1)).squeeze()  # Generate training data


        data_fig = plot_densities(y.detach().numpy(), x_lim=None, y_lim=None)
        data_fig
        plt.show()

        # Defining polynomial range
        #TODO: do I need span encreasing to better fit the bernstein polynomials?
        polynomial_range= torch.tensor([[y[:,0].min(),y[:,1].min()],
                                       [ y[:,0].max(), y[:,1].max()]], dtype=torch.float32)

        # Defining the model
        model = Decorrelation_Model(polynomial_range=polynomial_range,
                                                        number_variables=2,
                                                        degree_transformations=10,
                                                        span_restriction=None)

        model.forward(y, train=True)

        loss_list, number_iterations, \
        pen_value_ridge, pen_first_ridge, pen_second_ridge, \
        training_time, fig = train(model, y, penalty_params=torch.FloatTensor([0, 0, 0]),
                                   learning_rate=1, iterations=2000, verbose=False, patience=5,
                                   min_delta=1e-7, return_report=True)

        z_estimated = model.forward(y, train=False)
        latent_fig = plot_densities(z_estimated.detach().numpy(), x_lim=None, y_lim=None)
        latent_fig
        plt.show()

    def test_2(self):
        """
        span_restriction:"sigmoid"
        """

        loc = torch.zeros(2)
        lam = torch.Tensor([[1, 0],
                            [3, 1]])
        scale = lam @ torch.eye(2) @ torch.transpose(lam, 0, 1)
        y_distribution = MultivariateNormal(loc, scale)
        y = y_distribution.sample((2000,1)).squeeze()  # Generate training data


        data_fig = plot_densities(y.detach().numpy(), x_lim=None, y_lim=None)
        data_fig
        plt.show()

        # Defining polynomial range
        #TODO: do I need span encreasing to better fit the bernstein polynomials?
        polynomial_range= torch.tensor([[y[:,0].min(),y[:,1].min()],
                                       [ y[:,0].max(), y[:,1].max()]], dtype=torch.float32)

        # Defining the model
        model = Decorrelation_Model(polynomial_range=polynomial_range,
                                                        number_variables=2,
                                                        degree_transformations=10,
                                                        span_restriction="sigmoid")

        model.forward(y, train=True)

        loss_list, number_iterations, \
        pen_value_ridge, pen_first_ridge, pen_second_ridge, \
        training_time, fig = train(model, y, penalty_params=torch.FloatTensor([0, 0, 0]),
                                   learning_rate=1, iterations=2000, verbose=False, patience=5,
                                   min_delta=1e-7, return_report=True)

        z_estimated = model.forward(y, train=False)
        latent_fig = plot_densities(z_estimated.detach().numpy(), x_lim=None, y_lim=None)
        latent_fig
        plt.show()

    def test_3(self):
        """
        span_restriction:"reluler"
        """

        loc = torch.zeros(2)
        lam = torch.Tensor([[1, 0],
                            [3, 1]])
        scale = lam @ torch.eye(2) @ torch.transpose(lam, 0, 1)
        y_distribution = MultivariateNormal(loc, scale)
        y = y_distribution.sample((2000,1)).squeeze()  # Generate training data


        data_fig = plot_densities(y.detach().numpy(), x_lim=None, y_lim=None)
        data_fig
        plt.show()

        # Defining polynomial range
        #TODO: do I need span encreasing to better fit the bernstein polynomials?
        polynomial_range= torch.tensor([[y[:,0].min(),y[:,1].min()],
                                       [ y[:,0].max(), y[:,1].max()]], dtype=torch.float32)

        # Defining the model
        model = Decorrelation_Model(polynomial_range=polynomial_range,
                                                        number_variables=2,
                                                        degree_transformations=10,
                                                        span_restriction="reluler")

        model.forward(y, train=True)

        loss_list, number_iterations, \
        pen_value_ridge, pen_first_ridge, pen_second_ridge, \
        training_time, fig = train(model, y, penalty_params=torch.FloatTensor([0, 0, 0]),
                                   learning_rate=1, iterations=2000, verbose=False, patience=5,
                                   min_delta=1e-7, return_report=True)

        z_estimated = model.forward(y, train=False)
        latent_fig = plot_densities(z_estimated.detach().numpy(), x_lim=None, y_lim=None)
        latent_fig
        plt.show()



if __name__ == '__main__':
    unittest.main()