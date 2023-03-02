import mlflow
from run_simulation_study import run_simulation_study, log_mlflow_plot
import pandas as pd
import os
import numpy as np
from python_nf_mctm.simulation_study_helpers import *
from python_nf_mctm.hyperparameter_tuning_helpers import *


if __name__ == "__main__":

    run_info = mlflow.get_run(run_id= "c24d61ff214b4f2fb38a4bc51ee94253")

    model_uri = "runs:/{}/nf_mctm_model".format(run_info.info.run_id)
    loaded_model = mlflow.pytorch.load_model(model_uri)

    data_dims=3
    train_portion=0.8
    log_data=False

    complete_data = pd.DataFrame()
    for data_file in os.listdir("./Matthias_Epithel_sep"):
        data = pd.read_csv("./Matthias_Epithel_sep/" + data_file, index_col=0).T
        complete_data = complete_data.append(data, ignore_index=False)
    nonzerorow_data = complete_data.loc[~(complete_data.iloc[:, 3:] == 0).all(axis=1)]
    nonzerorow_gen_data = nonzerorow_data.iloc[:, 3:]
    genes_data_sub = nonzerorow_gen_data.iloc[:, 0:data_dims]
    genes_data_sub = genes_data_sub.loc[~(genes_data_sub == 0).all(axis=1)]
    genes_data_sub = np.array(genes_data_sub)
    if log_data:
        genes_data_sub = np.log(genes_data_sub + 1)
    # Splitting the data into train and test
    indices = np.random.permutation(genes_data_sub.shape[0])
    train_obs = int(np.round(genes_data_sub.shape[0] * train_portion))
    training_idx, test_idx = indices[:train_obs], indices[train_obs:]
    y_train, y_test = genes_data_sub[training_idx, :], genes_data_sub[test_idx, :]

    before = loaded_model.l1.params_inverse.clone()

    loaded_model.l1.approximate_inverse(input=y_train,
                                                              input_covariate=False,
                                                              spline_inverse="bspline",
                                                              degree_inverse=100,
                                                              monotonically_increasing_inverse=False,
                                                              iterations=1,
                                                              span_factor_inverse=0.1,
                                                              patience=20,
                                                              global_min_loss=0.001)

    #loaded_model.l1.forward(input=y_train, input_covariate=False, inverse=True)

    after = loaded_model.l1.params_inverse.clone()

    #print(before - after)

    plot_splines(layer=loaded_model.l1, y_train=torch.tensor(y_train),
                 covariate_exists=False)

    plt.show()


    x_sample = False
    y_sampled = loaded_model.sample(n_samples=5000, covariate=x_sample)
    y_sampled = y_sampled.detach().numpy()
    fig_y_sampled = plot_densities(y_sampled,
                                   covariate=x_sample)

    y_sampled_clip = y_sampled[np.sum(y_sampled > y_train.min(),1) == 3]
    y_sampled_clip = y_sampled_clip[np.sum(y_sampled_clip < y_train.max(),1) == 3]

    plot_densities(y_sampled_clip, covariate=x_sample)
    plt.show()









