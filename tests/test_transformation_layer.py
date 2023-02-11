import matplotlib.pyplot as plt
import unittest

from python_nf_mctm.training_helpers import *
from python_nf_mctm.bernstein_transformation_layer import *
from python_nf_mctm.simulation_study_helpers import *


class Multivariate_Transformation_Model(nn.Module):
    def __init__(self, polynomial_range, number_variables, degree_transformations=10, degree_decorrelation=10):
        super(Multivariate_Transformation_Model, self).__init__()
        self.polynomial_range = polynomial_range
        self.number_variables = number_variables

        self.degree_transformations = degree_transformations
        self.degree_decorrelation = degree_decorrelation

        self.l1 = Transformation(degree=self.degree_transformations, number_variables=self.number_variables,
                                 polynomial_range=self.polynomial_range.repeat(1, 3))

    def forward(self, y, train=True):
        if train:
            output, log_d = self.l1(y, return_log_d=True)
            return output, log_d, 0, 0, 0
        else:
            output = self.l1(y, return_log_d=False)
            return output


class TestFit(unittest.TestCase):

    def test_1(self):
        """
        Y Distribution: T

        --> Test simply needs to return a good looking latent space and some stable training plot
        """

        # Note: cannot set the scale to large as we then get huge outliers the model struggels with
        y_true = torch.distributions.studentT.StudentT(df=3, loc=5, scale=2).sample((1000, 2))

        data_fig = plot_densities(y_true.detach().numpy(), x_lim=None, y_lim=None)
        data_fig
        plt.show()

        # Defining polynomial range
        #TODO: do I need span encreasing to better fit the bernstein polynomials?
        polynomial_range= torch.tensor([[y_true[:,0].min()-1,y_true[:,1].min()-1],
                                       [ y_true[:,0].max()+1, y_true[:,1].max()+1]], dtype=torch.float32)

        # Defining the model
        model = Multivariate_Transformation_Model(polynomial_range=polynomial_range,
                                                        number_variables=2,
                                                        degree_transformations=10,
                                                        degree_decorrelation=10)

        model.forward(y_true, train=True)

        loss_list, number_iterations, \
        pen_value_ridge, pen_first_ridge, pen_second_ridge, \
        training_time, fig = train(model, y_true, penalty_params=torch.FloatTensor([0, 0, 0]),
                                   learning_rate=1, iterations=2000, verbose=False, patience=5,
                                   min_delta=1e-7, return_report=True)

        z_estimated = model.forward(y_true, train=False)
        latent_fig = plot_densities(z_estimated.detach().numpy(), x_lim=None, y_lim=None)
        latent_fig
        plt.show()


    def test_2(self):
        """
        Y Distribution: Gamma

        --> Test simply needs to return a good looking latent space and some stable training plot
        """

        # Note: cannot set the scale to large as we then get huge outliers the model struggels with
        y_true = torch.distributions.gamma.Gamma(concentration=2, rate=2).sample((1000, 2))

        data_fig = plot_densities(y_true.detach().numpy(), x_lim=None, y_lim=None)
        data_fig
        plt.show()

        # Defining polynomial range
        polynomial_range = torch.tensor([[y_true[:, 0].min() - 1, y_true[:, 1].min() - 1],
                                         [y_true[:, 0].max() + 1, y_true[:, 1].max() + 1]], dtype=torch.float32)

        # Defining the model
        model = Multivariate_Transformation_Model(polynomial_range=polynomial_range,
                                                  number_variables=2,
                                                  degree_transformations=10)

        model.forward(y_true, train=True)

        loss_list, number_iterations, \
        pen_value_ridge, pen_first_ridge, pen_second_ridge, \
        training_time, fig = train(model, y_true, penalty_params=torch.FloatTensor([0, 0, 0]),
                                   learning_rate=1, iterations=2000, verbose=False, patience=5,
                                   min_delta=1e-7, return_report=True)

        z_estimated = model.forward(y_true, train=False)
        latent_fig = plot_densities(z_estimated.detach().numpy(), x_lim=None, y_lim=None)
        latent_fig
        plt.show()

    def test_3(self):
        """
        Y Distribution: T
        Inverse Layer Training: True
        Monontonically Increasing Inverse: True
        """

        # Note: cannot set the scale to large as we then get huge outliers the model struggels with
        y_true = torch.distributions.studentT.StudentT(df=3, loc=5, scale=2).sample((1000, 2))

        data_fig = plot_densities(y_true.detach().numpy(), x_lim=None, y_lim=None)
        data_fig
        plt.show()

        # Defining polynomial range
        # TODO: do I need span encreasing to better fit the bernstein polynomials?
        polynomial_range = torch.tensor([[y_true[:, 0].min() - 1, y_true[:, 1].min() - 1],
                                         [y_true[:, 0].max() + 1, y_true[:, 1].max() + 1]], dtype=torch.float32)

        # Defining the model
        model = Multivariate_Transformation_Model(polynomial_range=polynomial_range,
                                                  number_variables=2,
                                                  degree_transformations=20)

        model.forward(y_true, train=True)

        loss_list, number_iterations, \
        pen_value_ridge, pen_first_ridge, pen_second_ridge, \
        training_time, fig = train(model, y_true, penalty_params=torch.FloatTensor([0, 0, 0]),
                                   learning_rate=1, iterations=10, verbose=False, patience=5,
                                   min_delta=1e-7, return_report=True)

        z_estimated = model.forward(y_true, train=False)
        latent_fig = plot_densities(z_estimated.detach().numpy(), x_lim=None, y_lim=None)
        latent_fig
        plt.show()

        #TODO: add an aprimate inverse functionality to nf_mctm
        #TODO: finish this test, the here defined model also needs an approximate inverse
        model.l1.approximate_inverse(y_true, iterations=300, monotonically_increasing_inverse=True, degree_inverse=100)
        plt.show()

        plot_splines(model.l1,y_train=y_true)
        plt.show()

        y_estimated = model.l1.forward(z_estimated, inverse=True)

        loss_abs = nn.L1Loss()
        self.assertTrue(loss_abs(y_estimated, y_true).item() < 0.5)


    def test_4(self):
        """
        Y Distribution: T
        Inverse Layer Training: True
        Monontonically Increasing Inverse: False
        """

        # Note: cannot set the scale to large as we then get huge outliers the model struggels with
        y_true = torch.distributions.studentT.StudentT(df=3, loc=5, scale=2).sample((1000, 2))

        data_fig = plot_densities(y_true.detach().numpy(), x_lim=None, y_lim=None)
        data_fig
        plt.show()

        # Defining polynomial range
        # TODO: do I need span encreasing to better fit the bernstein polynomials?
        polynomial_range = torch.tensor([[y_true[:, 0].min() - 1, y_true[:, 1].min() - 1],
                                         [y_true[:, 0].max() + 1, y_true[:, 1].max() + 1]], dtype=torch.float32)

        # Defining the model
        model = Multivariate_Transformation_Model(polynomial_range=polynomial_range,
                                                  number_variables=2,
                                                  degree_transformations=20)

        model.forward(y_true, train=True)

        loss_list, number_iterations, \
        pen_value_ridge, pen_first_ridge, pen_second_ridge, \
        training_time, fig = train(model, y_true, penalty_params=torch.FloatTensor([0, 0, 0]),
                                   learning_rate=1, iterations=10, verbose=False, patience=5,
                                   min_delta=1e-7, return_report=True)

        z_estimated = model.forward(y_true, train=False)
        latent_fig = plot_densities(z_estimated.detach().numpy(), x_lim=None, y_lim=None)
        latent_fig
        plt.show()

        #TODO: add an aprimate inverse functionality to nf_mctm
        #TODO: finish this test, the here defined model also needs an approximate inverse
        model.l1.approximate_inverse(y_true, iterations=300, monotonically_increasing_inverse=False)

        plot_splines(model.l1,y_train=y_true)
        plt.show()

        y_estimated = model.l1.forward(z_estimated, inverse=True)

        loss_abs = nn.L1Loss()
        self.assertTrue(loss_abs(y_estimated, y_true).item() < 0.5)


    def test_4(self):
        """
        Y Distribution: T
        Inverse Layer Training: True
        Spline: Bspline
        """

        # Note: cannot set the scale to large as we then get huge outliers the model struggels with
        y_true = torch.distributions.studentT.StudentT(df=3, loc=5, scale=2).sample((1000, 2))

        data_fig = plot_densities(y_true.detach().numpy(), x_lim=None, y_lim=None)
        data_fig
        plt.show()

        # Defining polynomial range
        # TODO: do I need span encreasing to better fit the bernstein polynomials?
        polynomial_range = torch.tensor([[y_true[:, 0].min() - 1, y_true[:, 1].min() - 1],
                                         [y_true[:, 0].max() + 1, y_true[:, 1].max() + 1]], dtype=torch.float32)

        # Defining the model
        model = Multivariate_Transformation_Model(polynomial_range=polynomial_range,
                                                  number_variables=2,
                                                  degree_transformations=20)

        model.forward(y_true, train=True)

        loss_list, number_iterations, \
        pen_value_ridge, pen_first_ridge, pen_second_ridge, \
        training_time, fig = train(model, y_true, penalty_params=torch.FloatTensor([0, 0, 0]),
                                   learning_rate=1, iterations=1, verbose=False, patience=5,
                                   min_delta=1e-7, return_report=True)

        z_estimated = model.forward(y_true, train=False)
        latent_fig = plot_densities(z_estimated.detach().numpy(), x_lim=None, y_lim=None)
        latent_fig
        plt.show()

        #TODO: add an aprimate inverse functionality to nf_mctm
        #TODO: finish this test, the here defined model also needs an approximate inverse
        model.l1.approximate_inverse(y_true, iterations=200,
                                     monotonically_increasing_inverse=True,
                                     spline_inverse="bspline",
                                     degree_inverse=40,
                                     lr=0.7,
                                     patience=20,
                                     span_factor_inv=0.2)
        plt.show()

        plot_splines(model.l1,y_train=y_true)
        plt.show()

        y_estimated = model.l1.forward(z_estimated, inverse=True)

        loss_abs = nn.L1Loss()
        self.assertTrue(loss_abs(y_estimated, y_true).item() < 0.5)



if __name__ == '__main__':
    unittest.main()