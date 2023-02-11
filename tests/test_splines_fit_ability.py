import torch
import torch.nn as nn
import numpy as np
import unittest
from python_nf_mctm.bernstein_prediction import bernstein_prediction
from python_nf_mctm.bspline_prediction import bspline_prediction
from python_nf_mctm.training_helpers import EarlyStopper
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from pytorch_lbfgs.LBFGS import FullBatchLBFGS

def fit_1d_spline(y_true, z_true, monotonically_increasing, global_min_loss, spline="bernstein"):

    # Fit Spline on the Data by
    # 1. Defining simple 1D Polynomial Model
    class spline_model(nn.Module):
        def __init__(self, degree, polynomial_range_inverse, params, spline="bernstein"):
            super().__init__()
            self.degree = degree
            self.polynomial_range_inverse = polynomial_range_inverse
            self.params = nn.Parameter(params, requires_grad=True)
            self.spline = spline

        def forward(self, x, monotonically_increasing=False):
            if self.spline == "bernstein":
                pred, a, b, c = bernstein_prediction(self.params,
                                                     x,
                                                     self.degree,
                                                     self.polynomial_range_inverse,
                                                     monotonically_increasing=monotonically_increasing,
                                                     derivativ=0)
            elif self.spline == "bspline":
                pred = bspline_prediction(self.params,
                                          x,
                                          self.degree,
                                          self.polynomial_range_inverse,
                                          monotonically_increasing=monotonically_increasing,
                                          derivativ=0)

            return pred

    # 2. Initializing the model with starting values
    span = z_true.max() - z_true.min()
    polynomial_range_inverse = torch.tensor([z_true.min() - 0.5 * span, z_true.max() + 0.5 * span], dtype=torch.float32)
    degree = 40
    params = torch.FloatTensor(np.repeat(0.1, degree + 1))

    spline_model = spline_model(degree=degree, polynomial_range_inverse=polynomial_range_inverse, params=params, spline=spline)

    y_estimated_init = spline_model.forward(z_true, monotonically_increasing=monotonically_increasing)

    sns.lineplot(x=z_true.detach().numpy(), y=y_true.detach().numpy())
    sns.lineplot(x=z_true.detach().numpy(), y=y_estimated_init.detach().numpy())
    plt.show()

    # 3. Fitting the model
    opt = FullBatchLBFGS(spline_model.parameters(), lr=1., history_size=1, line_search="Wolfe")#,
                            #max_iter=20,
                            #lr=0.1,
                            #history_size=1)
    #opt = torch.optim.Adam(spline_model.parameters(), lr=0.1)
    loss_fct = nn.L1Loss()
    patience = 1000
    min_delta = 1e-5
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, global_min_loss=global_min_loss)

    # print("initial loss is", loss_fct(y_estimated_init, y_true).item())

    def closure():
        opt.zero_grad()
        y_estimated = spline_model.forward(z_true, monotonically_increasing=monotonically_increasing)
        loss = loss_fct(y_estimated, y_true)  # use the `objective` function
        #loss.backward()  # backpropagate the loss # retain_graph=True
        #nn.utils.clip_grad_norm_(spline_model.parameters(), max_norm=1)
        #print(spline_model.params.grad)
        return loss

    loss = closure()
    loss.backward()
    options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}

    loss_list = []
    iterations = 1000
    for i in tqdm(range(iterations)):

        y_estimated = spline_model.forward(z_true, monotonically_increasing=monotonically_increasing)
        current_loss = loss_fct(y_estimated, y_true)
        loss_list.append(current_loss.detach().numpy().item())

        #loss = closure()
        #loss.backward()

        opt.step(options)

        if early_stopper.early_stop(current_loss.detach().numpy(), spline_model):
            print("Early Stop at iteration", i, "with loss", current_loss.item(), "and patience", patience,
                  "and min_delta", min_delta)
            break

    spline_model.load_state_dict(early_stopper.best_model_state)

    print("Final loss", current_loss.item())

    with sns.axes_style('ticks'):
        plt.plot(loss_list)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
    sns.despine(trim=True)
    plt.show()

    sns.lineplot(x=z_true.detach().numpy(), y=y_true.detach().numpy())
    sns.lineplot(x=z_true.detach().numpy(), y=y_estimated.detach().numpy())
    plt.show()

    y_estimated = spline_model.forward(z_true, monotonically_increasing=monotonically_increasing)

    return y_estimated


class TestFit(unittest.TestCase):
    """
    Tests that Bernstein polynomial and BSpline fitting works

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    All the takeaways are correct for the vanilla LBFGS implementation in Pytorch
    Using the better one from https://github.com/hjmshi/PyTorch-LBFGS with FullBatch LBFGS solves the problems
    --> apparently the solution was that the LBFGS from the package had line_search="Wolfe" by default and that worked well
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    def test_1(self):
        """
        Spline: Bernstein
        Fct: simple shifted curve

        Takeaway:
        - Fitting is more accurate when setting monotonically_increasing to false
            - Mono:False attains 0.01 loss after 5 iterations
            - Mono:True attains 0.034 loss after 10 iterations and does not improve notably after, even worsens to 0.07 if run for 1000 iterations
        - Monotonically increasing is inaccurate at the top
        - Both are very fast
        """
        monotonically_increasing=False
        global_min_loss = 0.01

        # Training Data
        z_true = torch.linspace(-5, 5, 1000)
        y_true = z_true*2 + 2

        y_estimated = fit_1d_spline(y_true, z_true, monotonically_increasing, global_min_loss)
        loss_abs = nn.L1Loss()
        self.assertTrue(loss_abs(y_estimated, y_true).item() < global_min_loss)

    def test_2(self):
        """
        Spline: Bernstein
        Fct: taylor series

        Takeaway:
        - Spline struggles to fit the beginning of the curve, it starts as a  flat line while data is steep
        - This cannot be improved by widening the polynomial range or increasing the degree
        - This fit becomes perfect very quickly once one sets monotonically_increasing to False
            - Mono:False attains 0.05 loss after 6 iterations (cannot get to 0.01 as loss blows up after 13 iterations)
            - Mono:True attains 0.19 loss after 397 iterations and when running 1000 Iterations it cannot get better than 0.155
        """
        monotonically_increasing=False
        global_min_loss = 0.01

        # Training Data
        z_true = torch.linspace(-5, 5, 1000)
        y_true = 2 + 2 * z_true + 0.05 * z_true**2 + 0.2 * z_true**3

        y_estimated = fit_1d_spline(y_true, z_true, monotonically_increasing, global_min_loss)
        loss_abs = nn.L1Loss()
        self.assertTrue(loss_abs(y_estimated, y_true).item() < global_min_loss)

    def test_3(self):
        """
        Spline: Bernstein
        Fct: sigmoid

        Takeaway:
        - monotonically_increasing=True is very fast and accurate as it gets to 0.01 loss after 95 iterations
        - monotonically_increasing=False explodes in its loss after iteration 18 and cannot train stable,  even with lr=0.01
        """
        monotonically_increasing=True
        global_min_loss = 0.01

        # Training Data
        z_true = torch.linspace(-5, 5, 1000)
        y_true = torch.sigmoid(z_true)

        y_estimated = fit_1d_spline(y_true, z_true, monotonically_increasing, global_min_loss)
        loss_abs = nn.L1Loss()
        self.assertTrue(loss_abs(y_estimated, y_true).item() < global_min_loss)

    def test_4(self):
        """
        Spline: BSpline
        Fct: simple shifted curve

        Takeaway:
        - apparently the knot span matters a lot here, using LBFGS optimizer
            - works well for: 0.2,0.3 (okay fit for 0.1)
            - fails for: 0.25 and 0.5 as well as 0 (no adjustment)
            - I checked if the issue is that one knot lies perfectly on 0.0 but that does not explan for what span adjustments it fails
        - bspline train fast and accurate with adam optimizer
            - does require some larger polynomial range to fit borders well
            - once we increase by 0.1 it fits perfectly, also works for 0.25 where LBFGS failed for example

        """
        spline = "bspline"
        monotonically_increasing=False
        global_min_loss = 0.01

        # Training Data
        z_true = torch.linspace(-5, 5, 1000)
        y_true = z_true*2 + 2

        y_estimated = fit_1d_spline(y_true, z_true, monotonically_increasing, global_min_loss, spline)
        loss_abs = nn.L1Loss()
        self.assertTrue(loss_abs(y_estimated, y_true).item() < global_min_loss)


if __name__ == '__main__':
    unittest.main()

