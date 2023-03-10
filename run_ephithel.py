import mlflow
from run_simulation_study import run_simulation_study, log_mlflow_plot
import pandas as pd
import os
import numpy as np
from python_nf_mctm.simulation_study_helpers import *
from python_nf_mctm.hyperparameter_tuning_helpers import *

def run_epithel_study(data_dims: int,
                      train_portion: float,
                      val_portion: float,
                      experiment_id: int,
                      log_data: bool,
                      # Setting Hyperparameter Values
                      seed_value: int,
                      penvalueridge_list: list,
                      penfirstridge_list: list,
                      pensecondridge_list: list,
                      poly_span_abs: float,
                      spline_decorrelation: str,
                      spline_inverse: str,
                      iterations: int,
                      iterations_inverse: int,
                      learning_rate_list: list,
                      patience_list: list,
                      min_delta_list: list,
                      degree_transformations_list: list,
                      degree_decorrelation_list: list,
                      lambda_penalty_params_list: list = False,
                      monotonically_increasing_inverse: bool = True,
                      span_factor: float = 0.1,
                      span_factor_inverse: float = 0.1,
                      span_restriction: str = None,
                      degree_inverse: int = 0,
                      hyperparameter_tuning: bool = True,
                      cross_validation_folds: int = False,
                      tune_precision_matrix_penalty: bool = False,
                      iterations_hyperparameter_tuning: int = 1500,
                      n_samples: int = 2000,
                      list_comprehension: bool = False,
                      optimizer: str = "LBFGS",
                      num_decorr_layers: int = 6,
                      spline_transformation="bernstein",
                      device=None):

    covariate_exists=False
    number_covariates = 0
    x_train=False
    x_test=False
    x_validate=False

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
    train_obs = int(np.round(genes_data_sub.shape[0]*train_portion))
    training_idx, test_idx = indices[:train_obs], indices[train_obs:]
    y_train, y_test = genes_data_sub[training_idx, :], genes_data_sub[test_idx, :]

    # Starting the MLflow run
    mlflow.start_run(
        run_name="{}".format(seed_value),
        experiment_id=experiment_id,
        tags={"seed": seed_value, "data_dims": data_dims}
    )
    print("Started Run")


    if hyperparameter_tuning:

        # Splitting the data into sub_train and validation
        indices = np.random.permutation(y_train.shape[0])
        sub_train_obs = int(np.round(y_train.shape[0] * (1-val_portion)))
        training_idx, test_idx = indices[:sub_train_obs], indices[sub_train_obs:]
        y_sub_train, y_validate = y_train[training_idx, :], y_train[test_idx, :]

        y_sub_train = torch.tensor(y_sub_train, dtype=torch.float32)#, device=device)
        y_validate = torch.tensor(y_validate, dtype=torch.float32)#, device=device)

        # Running Cross validation to identify hyperparameters
        results = run_hyperparameter_tuning(y_sub_train,
                                            y_validate,
                                            poly_span_abs, iterations_hyperparameter_tuning, spline_decorrelation,
                                        penvalueridge_list, penfirstridge_list, pensecondridge_list,
                                        lambda_penalty_params_list,
                                        learning_rate_list, patience_list, min_delta_list,
                                        degree_transformations_list, degree_decorrelation_list,
                                            x_train = x_train,
                                            x_validate = x_validate,
                                            tune_precision_matrix_penalty=tune_precision_matrix_penalty,
                                            cross_validation_folds=cross_validation_folds,
                                            device=device) #normalisation_layer_list

        fig_hyperparameter_tuning_cooordinate = optuna.visualization.plot_parallel_coordinate(results)
        fig_hyperparameter_tuning_contour = optuna.visualization.plot_contour(results)
        fig_hyperparameter_tuning_slice = optuna.visualization.plot_slice(results)
        fig_hyperparameter_tuning_plot_param_importances = optuna.visualization.plot_param_importances(results)
        fig_hyperparameter_tuning_edf = optuna.visualization.plot_edf(results)
        #TODO: store the hyperparameter tuning figures as artifacts

        log_mlflow_plot(fig_hyperparameter_tuning_cooordinate, "fig_hyperparameter_tuning_cooordinate.html", type="plotly")
        log_mlflow_plot(fig_hyperparameter_tuning_contour, "fig_hyperparameter_tuning_cooordinate.html", type="plotly")
        log_mlflow_plot(fig_hyperparameter_tuning_slice, "fig_hyperparameter_tuning_slice.html", type="plotly")
        log_mlflow_plot(fig_hyperparameter_tuning_plot_param_importances, "fig_hyperparameter_tuning_plot_param_importances.html", type="plotly")
        log_mlflow_plot(fig_hyperparameter_tuning_edf, "fig_hyperparameter_tuning_edf.html", type="plotly")


        #optimal_hyperparameters, results_summary = extract_optimal_hyperparameters(results)
        #penvalueridge, penfirstridge, pensecondridge, learning_rate, \
        #patience, min_delta, degree_transformations, \
        #degree_decorrelation  = optimal_hyperparameters #normalisation_layer

        penvalueridge = penvalueridge_list[0]
        penfirstridge = results.best_params["penfirstridge"]
        pensecondridge = results.best_params["pensecondridge"]

        lambda_penalty_params = lambda_penalty_params_list[0]
        learning_rate = learning_rate_list[0]
        patience = patience_list[0]
        min_delta = min_delta_list[0]
        degree_transformations = degree_transformations_list[0]
        degree_decorrelation = degree_decorrelation_list[0]
    else:
        # Setting Hyperparameter Values
        penvalueridge = penvalueridge_list[0]
        penfirstridge = penfirstridge_list[0]
        pensecondridge = pensecondridge_list[0]
        lambda_penalty_params = lambda_penalty_params_list[0]
        learning_rate = learning_rate_list[0]
        patience = patience_list[0]
        min_delta = min_delta_list[0]
        degree_transformations = degree_transformations_list[0]
        degree_decorrelation = degree_decorrelation_list[0]
        #normalisation_layer = normalisation_layer_list[0]

    mlflow.log_param(key="train_portion", value=train_portion)
    mlflow.log_param(key="val_portion", value=val_portion)
    mlflow.log_param(key="pen_value_ridge", value=penvalueridge)
    mlflow.log_param(key="pen_first_ridge", value=penfirstridge)
    mlflow.log_param(key="pen_second_ridge", value=pensecondridge)
    mlflow.log_param(key="poly_span_abs", value=poly_span_abs)
    mlflow.log_param(key="spline_decorrelation", value=spline_decorrelation)
    mlflow.log_param(key="iterations", value=iterations)
    mlflow.log_param(key="iterations_inverse", value=iterations_inverse)
    mlflow.log_param(key="monotonically_increasing_inverse", value=monotonically_increasing_inverse)
    mlflow.log_param(key="span_factor", value=span_factor)
    mlflow.log_param(key="span_factor_inverse", value=span_factor_inverse)
    mlflow.log_param(key="span_restriction", value=span_restriction)
    mlflow.log_param(key="degree_inverse", value=degree_inverse)
    mlflow.log_param(key="hyperparameter_tuning", value=hyperparameter_tuning)
    mlflow.log_param(key="learning_rate", value=learning_rate)
    mlflow.log_param(key="patience", value=patience)
    mlflow.log_param(key="min_delta", value=min_delta)
    mlflow.log_param(key="degree_transformations", value=degree_transformations)
    mlflow.log_param(key="degree_decorrelation", value=degree_decorrelation)
    mlflow.log_param(key="spline_inverse", value=spline_inverse)
    mlflow.log_param(key="iterations_hyperparameter_tuning", value=iterations_hyperparameter_tuning)
    mlflow.log_param(key="covariate_exists", value=covariate_exists)
    mlflow.log_param(key="list_comprehension", value=list_comprehension)
    mlflow.log_param(key="optimizer", value=optimizer)
    mlflow.log_param(key="log_data", value=log_data)
    mlflow.log_param(key="num_decorr_layers", value=num_decorr_layers)
    mlflow.log_param(key="spline_transformation", value=spline_transformation)
    mlflow.log_param(key="cross_validation_folds", value=cross_validation_folds)

    # Defining the model
    poly_range = torch.FloatTensor([[-poly_span_abs], [poly_span_abs]])#.to(device)
    penalty_params = torch.tensor([penvalueridge,
                                   penfirstridge,
                                   pensecondridge])#.to(device)

    # comes out of CV aas nan which then doe
    # normalisation_layer = None

    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    # Splitting the data into sub_train and validation
    indices = np.random.permutation(y_train.shape[0])
    sub_train_obs = int(np.round(y_train.shape[0] * (1 - val_portion)))
    training_idx, test_idx = indices[:sub_train_obs], indices[sub_train_obs:]
    y_sub_train, y_validate = y_train[training_idx, :], y_train[test_idx, :]

    y_sub_train = torch.tensor(y_sub_train, dtype=torch.float32, device=device)
    y_validate = torch.tensor(y_validate, dtype=torch.float32, device=device)

    nf_mctm = NF_MCTM(input_min=y_sub_train.min(0).values,
                      input_max=y_sub_train.max(0).values,
                      polynomial_range=poly_range,
                      number_variables=y_sub_train.size()[1],
                      spline_transformation=spline_transformation,
                      spline_decorrelation=spline_decorrelation,
                      degree_transformations=int(degree_transformations),
                      degree_decorrelation=int(degree_decorrelation),
                      span_factor=span_factor,
                      span_restriction=span_restriction,
                      number_covariates=number_covariates,
                      list_comprehension=list_comprehension,
                      num_decorr_layers=num_decorr_layers,
                      device=device)

    #parallelizing over multiple GPUs
    #nf_mctm = nn.DataParallel(nf_mctm)

    nf_mctm = nf_mctm.to(device)
    # normalisation_layer=normalisation_layer)

    # Training the model
    loss_training_iterations, number_iterations, pen_value_ridge_final, pen_first_ridge_final, pen_second_ridge_final, \
    pen_lambda_lasso, training_time, fig_training = train(model=nf_mctm,
                                                          train_data=y_sub_train,
                                                          validate_data=y_validate,
                                                          train_covariates=False,
                                                          validate_covariates=False,
                                                          penalty_params=penalty_params,
                                                          lambda_penalty_params=lambda_penalty_params,
                                                          iterations=iterations,
                                                          learning_rate=learning_rate,
                                                          patience=patience,
                                                          min_delta=min_delta,
                                                          verbose=False,
                                                          optimizer=optimizer)

    y_sub_train = y_sub_train.cpu()
    nf_mctm = nf_mctm.cpu()
    nf_mctm.device = None

    # Training the inverse of the model
    fig_training_inverse = nf_mctm.transformation.approximate_inverse(input=y_sub_train,
                                                          input_covariate=x_train,
                                                          spline_inverse=spline_inverse,
                                                          degree_inverse=degree_inverse,
                                                          monotonically_increasing_inverse=monotonically_increasing_inverse,
                                                          iterations=iterations_inverse,
                                                          span_factor_inverse=span_factor_inverse,
                                                          patience=20,
                                                          global_min_loss=0.001)

    #### Training Evaluation

    #sum_log_likelihood_val = nf_mctm.log_likelihood(y_validate, x_validate).detach().numpy().sum()

    fig_y_train = plot_densities(y_train, covariate=x_train, x_lim=[y_train.min(), y_train.max()],
                                 y_lim=[y_train.min(), y_train.max()])

    fig_splines_transformation_layer = plot_splines(layer=nf_mctm.transformation, y_train=y_train,
                                                      covariate_exists=covariate_exists)
    log_mlflow_plot(fig_splines_transformation_layer, 'plot_splines_transformation_layer.png')

    for i in range(nf_mctm.number_decorrelation_layers):
        fig_splines_decorrelation_layer = plot_splines(layer=nf_mctm.decorrelation_layers[i], y_train=y_train, covariate_exists=covariate_exists)
        log_mlflow_plot(fig_splines_decorrelation_layer, 'plot_splines_decorrelation_layer_{}.png'.format(i))

    #fig_splines_decorrelation_layer_2 = plot_splines(layer=nf_mctm.l2, covariate_exists=covariate_exists)
    #fig_splines_decorrelation_layer_4 = plot_splines(layer=nf_mctm.l4, covariate_exists=covariate_exists)
    #fig_splines_decorrelation_layer_6 = plot_splines(layer=nf_mctm.l6, covariate_exists=covariate_exists)

    # Evaluate latent space of the model in training set
    z_train = nf_mctm.latent_space_representation(y_train, x_train).detach().numpy()
    res_normal_train, res_pval_train, z_mean_train, z_cov_train, p_train = evaluate_latent_space(z_train)
    fig_z_train = plot_densities(z_train, x_train)

    predicted_train_log_likelihood = nf_mctm.log_likelihood(y_train, x_train)

    precision_matrix_train = nf_mctm.compute_precision_matrix(y_train, x_train)
    fig_precision_matrix_nf_mctm_train = plot_metric_scatter(y_train, precision_matrix_train,
                                                             metric_type="precision_matrix")
    fig_hist_precision_matrix_nf_mctm_train = plot_metric_hist(precision_matrix_train, covariate=False)

    # estimate the Multivariate Normal Distribution as Model
    mean_mvn_model = y_train.mean(0)  # 0 to do mean across dim 0 not globally
    cov_mvn_model = y_train.T.cov()
    mvn_model = MultivariateNormal(loc=mean_mvn_model, covariance_matrix=cov_mvn_model)

    train_log_likelihood_mvn_model = mvn_model.log_prob(y_train)

    #### Test Evaluation

    fig_y_test = plot_densities(y_test, covariate=x_test)

    # Evaluate latent space of the model in test set
    z_test = nf_mctm.latent_space_representation(y_test, x_test).detach().numpy()
    res_normal_test, res_pval_test, z_mean_test, z_cov_test, p_test = evaluate_latent_space(z_test)
    fig_z_test = plot_densities(z_test)

    predicted_test_log_likelihood = nf_mctm.log_likelihood(y_test, x_test)

    # mvn Model on test data
    test_log_likelihood_mvn_model = mvn_model.log_prob(y_test)

    #### Log Training Artifacts
    model_info = mlflow.pytorch.log_model(nf_mctm, "nf_mctm_model")

    log_mlflow_plot(fig_training, 'plot_training.png')
    #log_mlflow_plot(fig_training_inverse, 'plot_training_inverse.png')

    # fig_training.savefig('plot_training.png')
    # mlflow.log_artifact("./plot_training.png")
    # fig_training_inverse.savefig('plot_training_inverse.png')
    # mlflow.log_artifact("./plot_training_inverse.png")
    mlflow.log_metric("number_iterations", number_iterations)
    mlflow.log_metric("training_time", training_time)
    #mlflow.log_metric("sum_log_likelihood_validation", sum_log_likelihood_val)

    mlflow.log_metric("pen_value_ridge_final", pen_value_ridge_final)
    mlflow.log_metric("pen_first_ridge_final", pen_first_ridge_final)
    mlflow.log_metric("pen_second_ridge_final", pen_second_ridge_final)

    np.save("loss_training_iterations.npy", loss_training_iterations)
    mlflow.log_artifact("./loss_training_iterations.npy")

    #log_mlflow_plot(fig_splines_transformation_layer_1, 'plot_splines_transformation_layer_1.png')
    #log_mlflow_plot(fig_splines_decorrelation_layer_2, 'plot_splines_decorrelation_layer_2.png')
    #log_mlflow_plot(fig_splines_decorrelation_layer_4, 'plot_splines_decorrelation_layer_4.png')
    #log_mlflow_plot(fig_splines_decorrelation_layer_6, 'plot_splines_decorrelation_layer_6.png')

    log_mlflow_plot(fig_precision_matrix_nf_mctm_train, 'plot_precision_matrix_nf_mctm_train.png')
    log_mlflow_plot(fig_hist_precision_matrix_nf_mctm_train, 'plot_hist_precision_matrix_nf_mctm_train.png')

    np.save("precision_matrix_train.npy", precision_matrix_train)
    mlflow.log_artifact("./precision_matrix_train.npy")

    #### Log Train Data Metrics and Artifacts
    log_mlflow_plot(fig_y_train, 'plot_data_train.png')
    # fig_y_train.savefig('plot_data_train.png')
    # mlflow.log_artifact("./plot_data_train.png")

    mlflow.log_metric("mv_normality_result_train", res_normal_train)
    mlflow.log_metric("mv_normality_pval_train", res_pval_train)

    np.save("z_mean_train.npy", z_mean_train)
    mlflow.log_artifact("./z_mean_train.npy")

    np.save("z_cov_train.npy", z_cov_train)
    mlflow.log_artifact("./z_cov_train.npy")

    np.save("uv_normality_pvals_train.npy", p_train)
    mlflow.log_artifact("./uv_normality_pvals_train.npy")

    log_mlflow_plot(fig_z_train, 'plot_latent_space_train.png')
    # fig_z_train.savefig('plot_latent_space_train.png')
    # mlflow.log_artifact("./plot_latent_space_train.png")

    mlflow.log_metric("log_likelihood_nf_mctm_train", predicted_train_log_likelihood.sum())
    mlflow.log_metric("log_likelihood_mvn_model_train", train_log_likelihood_mvn_model.sum())

    #### Log Test Data Metrics and Artifacts
    log_mlflow_plot(fig_y_test, 'plot_data_test.png')
    # fig_y_test.savefig('plot_data_test.png')
    # mlflow.log_artifact("./plot_data_test.png")

    mlflow.log_metric("mv_normality_result_test", res_normal_test)
    mlflow.log_metric("mv_normality_pval_test", res_pval_test)

    np.save("z_mean_test.npy", z_mean_test)
    mlflow.log_artifact("./z_mean_test.npy")

    np.save("z_cov_test.npy", z_cov_test)
    mlflow.log_artifact("./z_cov_test.npy")

    np.save("uv_normality_pvals_test.npy", p_test)
    mlflow.log_artifact("./uv_normality_pvals_test.npy")

    log_mlflow_plot(fig_z_test, 'plot_latent_space_test.png')
    # fig_z_test.savefig('plot_latent_space_test.png')
    # mlflow.log_artifact("./plot_latent_space_test.png")

    mlflow.log_metric("log_likelihood_nf_mctm_test", predicted_test_log_likelihood.sum())
    mlflow.log_metric("log_likelihood_mvn_model_test", test_log_likelihood_mvn_model.sum())

    x_sample = False
    if n_samples == "train_samples":
        n_samples = len(y_train)

    y_sampled = nf_mctm.sample(n_samples=n_samples, covariate=x_sample)
    y_sampled = y_sampled.detach().numpy()
    fig_y_sampled = plot_densities(y_sampled,
                                   covariate=x_sample)

    #### Log Sampling Aritfacts
    np.save("synthetically_sampled_data.npy", y_sampled)
    mlflow.log_artifact("./synthetically_sampled_data.npy")

    log_mlflow_plot(fig_y_sampled, 'plot_synthetically_sampled_data.png')

    #End the run
    print("Finished Run")
    mlflow.end_run()


if __name__ == "__main__":

    import torch

    if torch.cuda.is_available():
        device = "cuda:0"
        print("running on cuda")
        print("number of gpus:",torch.cuda.device_count())
    else:
        device = "cpu"
        print("running on cpu")

        #mlflow.create_experiment(name="ephithel_hyperpar_server2")
    experiment = mlflow.get_experiment_by_name("ephithel_hyperpar_server2")

    run_epithel_study(
            experiment_id = experiment.experiment_id,
            data_dims=3,
            train_portion=0.8,
            val_portion=0.3,
            log_data=False,
            # Setting Hyperparameter Values
            seed_value=1,
            penvalueridge_list=[0],
            penfirstridge_list=[0.021847138083227073],
            pensecondridge_list=[0.04247392961503296],
            poly_span_abs=15,
            spline_transformation="bspline",
            spline_decorrelation="bspline",
            spline_inverse="bspline",
            span_factor=0.1,
            span_factor_inverse=0.2,
            span_restriction="reluler",
            iterations=50,
            iterations_hyperparameter_tuning=50,
            iterations_inverse=1,
            learning_rate_list=[1.],
            patience_list=[5],
            min_delta_list=[1e-8],
            degree_transformations_list=[70],
            degree_decorrelation_list=[40],
            lambda_penalty_params_list=[False],
            #normalisation_layer_list=[None],
            degree_inverse=120,
            monotonically_increasing_inverse=True,
            hyperparameter_tuning=True,
            cross_validation_folds=5,
            n_samples=8000,
            list_comprehension=True,
            num_decorr_layers=6,
            device=device)

    #run_epithel_study(
    #    experiment_id= experiment.experiment_id,
    #    data_dims=3,
    #    train_portion=0.8,
    #    val_portion=0.3,
    #    log_data=True,
    #    # Setting Hyperparameter Values
    #    seed_value=1,
    #    penvalueridge_list=[0],
    #    penfirstridge_list=[0],
    #    pensecondridge_list=[0],
    #    poly_span_abs=15,
    #    spline_decorrelation="bspline",
    #    spline_inverse="bspline",
    #    span_factor=0.1,
    #    span_factor_inverse=0.2,
    #    span_restriction="reluler",
    #    iterations=10,
    #    iterations_hyperparameter_tuning=10,
    #    iterations_inverse=1,
    #    learning_rate_list=[1.],
    #    patience_list=[10],
    #    min_delta_list=[1e-8],
    #    degree_transformations_list=[70],
    #    degree_decorrelation_list=[40],
    #    lambda_penalty_params_list=[False],
    #    # normalisation_layer_list=[None],
    #    degree_inverse=120,
    #    monotonically_increasing_inverse=True,
    #    hyperparameter_tuning=True,
    #    n_samples=10000,
    #    list_comprehension=True,
    #    num_decorr_layers=6)