from python_nf_mctm.bernstein_transformation_layer import bernstein_prediction, compute_starting_values
import torch
import torch.nn as nn
from python_nf_mctm.training_helpers import EarlyStopper
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # Define a spline and pass some test data through it
    y_true = torch.linspace(-5, 5, 100)
    polynomial_range = torch.tensor([-5,5])
    degree=10
    params = compute_starting_values(degree,
                                     polynomial_range[0],
                                     polynomial_range[1],
                                     1)
    z_true,a,b,c = bernstein_prediction(params.flatten(),
                                  y_true,
                                  degree,
                                  polynomial_range,
                                  monotonically_increasing=True,
                                  derivativ=0)

    sns.lineplot(x=z_true, y=y_true)
    plt.show()

    # Fit reverse spline on the data
    polynomial_range_inverse = torch.tensor([z_true.min(),z_true.max()], dtype=torch.float32)
    degree=20
    params = torch.FloatTensor(np.repeat(0.1,degree+1))

    class spline_model(nn.Module):
        def __init__(self,degree,polynomial_range_inverse,params):
            super().__init__()
            self.degree = degree
            self.polynomial_range_inverse = polynomial_range_inverse
            self.params = nn.Parameter(params, requires_grad=True)

        def forward(self, x, monotonically_increasing=False):


            #pred = bspline_prediction(self.params,
            #                          x,
            #                          self.degree,
            #                          self.polynomial_range_inverse,
            #                          monotonically_increasing=monotonically_increasing,
            #                          derivativ=0)

            pred, a,b,c = bernstein_prediction(self.params,
                                      x,
                                      self.degree,
                                      self.polynomial_range_inverse,
                                      monotonically_increasing=monotonically_increasing,
                                      derivativ=0)

            return pred

    spline_model = spline_model(degree=degree,polynomial_range_inverse=polynomial_range_inverse,params=params)

    y_estimated_init = spline_model.forward(z_true, monotonically_increasing=False)

    sns.lineplot(x=z_true.detach().numpy(), y=y_true.detach().numpy())
    sns.lineplot(x=z_true.detach().numpy(), y=y_estimated_init.detach().numpy())
    plt.show()

    opt = torch.optim.LBFGS(spline_model.parameters(), lr=0.01, history_size=1)
    loss_fct = nn.L1Loss()
    patience = 100
    min_delta = 1e-5
    global_min_loss = 0.0001
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, global_min_loss=global_min_loss)

    print("initial loss is", loss_fct(y_estimated_init, y_true).item())

    def closure():
        opt.zero_grad()
        y_estimated = spline_model.forward(z_true, monotonically_increasing=True)
        loss = loss_fct(y_estimated, y_true)  # use the `objective` function
        loss.backward()  # backpropagate the loss # retain_graph=True
        return loss


    loss_list = []
    iterations=1000
    for i in tqdm(range(iterations)):


        y_estimated  = spline_model.forward(z_true, monotonically_increasing=True)
        current_loss = loss_fct(y_estimated, y_true)
        loss_list.append(current_loss.detach().numpy().item())

        opt.step(closure)

        if early_stopper.early_stop(current_loss.detach().numpy()):
            print("Early Stop at iteration", i, "with loss", current_loss.item(), "and patience", patience,
                  "and min_delta", min_delta)
            break


    print("Final loss", current_loss.item())


    with sns.axes_style('ticks'):s
        plt.plot(loss_list)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
    sns.despine(trim = True)
    plt.show()

    sns.lineplot(x=z_true.detach().numpy(), y=y_true.detach().numpy())
    sns.lineplot(x=z_true.detach().numpy(), y=y_estimated.detach().numpy())
    plt.show()

    print("end")

