import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import mlflow

from simulation_study_helpers import *
from training_helpers import *
from nf_mctm import *
from hyperparameter_tuning_helpers import *

if __name__ == "__main__":
    experiment_data = mlflow.search_runs(experiment_ids=["2"])

    run_id = experiment_data["run_id"][0]

    run = mlflow.get_run(run_id)
    seed_value = run.data.tags["seed"]
    copula = run.data.tags["copula"]
    copula_par = run.data.tags["copula_par"]
    train_obs = run.data.tags["train_obs"]

    experiment_folder = str(copula) + "_" + str(copula_par) + "_" + str(train_obs) + "/"
    y_train = torch.tensor(
        pd.read_csv("simulation_study_data/" + experiment_folder + str(seed_value) + "_sample_train.csv").values,
        dtype=torch.float32)

    model = mlflow.pytorch.load_model("runs:/" + run_id + "/nf_mctm_model")

    #z_tilde = model.l1.forward(y_train)

    # Defining the model
    poly_range = torch.FloatTensor([[-5], [5]])
    penalty_params = torch.tensor([0,
                                   0,
                                   0])

    #model = NF_MCTM(input_min=y_train.min(0).values,
    #                  input_max=y_train.max(0).values,
    #                  polynomial_range=poly_range,
    #                  number_variables=y_train.size()[1],
    #                  spline_decorrelation="bspline",
    #                  degree_transformations=int(10),
    #                  degree_decorrelation=int(10))
    model.l1.spline = "bernstein"
    model.l1.approximate_inverse(input=y_train, iterations=200, lr=0.001, inv_degree=40, inv_spline="bernstein")

    #z_tilde = model.l1.forward(y_train, monotonically_increasing = True)
    #y_est = model.l1.forward(z_tilde,inverse=True, monotonically_increasing = True)
#
    #z_tilde = z_tilde.detach().numpy()
    #y_est = y_est.detach().numpy()
#
    #sns.lineplot(x=z_tilde[:, 0], y=y_train[:, 0])
    #sns.lineplot(x=z_tilde[:, 1], y=y_train[:, 1])
#
    #inv_error=y_est-z_tilde
    #plt.hist(inv_error[:, 0])
    #plt.hist(inv_error[:, 1])

    #plot_splines(model.l1)
    #plt.show()

    #model.l1.params_inverse

    #input_space = torch.vstack([torch.linspace(y_train[:, 0].min(), y_train[:, 0].max(), 2000),
    #                            torch.linspace(y_train[:, 1].min(), y_train[:, 1].max(), 2000)]).T
    #y_train = input_space

    z_tilde = model.l1.forward(y_train, monotonically_increasing=True)
    y_estimated = model.l1.forward(z_tilde, inverse=True, monotonically_increasing=True)
    num_splines=2
    layer = model.l1

    y_train = y_train.detach().numpy()
    z_tilde = z_tilde.detach().numpy()
    y_estimated = y_estimated.detach().numpy()
    fig, axs = plt.subplots(nrows=1, ncols=num_splines, figsize=(15, 5),
                            gridspec_kw={'wspace': 0.0, 'hspace': 0.0}, sharey=True)
    a=0
    for spline_num in range(num_splines):
        sns.lineplot(x=y_train[:, spline_num], y=z_tilde[:, spline_num], ax=axs[a])
        if layer.type == "transformation":
            # sns.lineplot(x=y_train[:,spline_num], y=output_derivativ[:,spline_num], ax = axs[a])
            sns.lineplot(x=y_estimated[:, spline_num], y=z_tilde[:, spline_num], ax=axs[a])
        axs[a].set_ylim(z_tilde.min(), z_tilde.max())
        axs[a].set_xlim(y_train[:, spline_num].min(), y_train[:, spline_num].max())
        a += 1
    plt.show()

    plot_splines(model.l1)

    print("Done!")