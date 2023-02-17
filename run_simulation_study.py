import pandas as pd
import mlflow
import torch

from python_nf_mctm.simulation_study_helpers import *
from python_nf_mctm.hyperparameter_tuning_helpers import *

def log_mlflow_plot(fig,file_name,type="plt"):
    if type == "plt":
        fig.savefig(file_name)
    elif type == "plotly":
        fig.write_html(file_name)

    mlflow.log_artifact("./"+file_name)

def run_simulation_study(
        experiment_id: int,
        copula: str,
        copula_par: float,
        train_obs: int,
        covariate_exists: str,
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
        #normalisation_layer_list: list,
        lambda_penalty_params_list: list = False,
        monotonically_increasing_inverse: bool = True,
        span_factor: float = 0.1,
        span_factor_inverse: float = 0.1,
        span_restriction: str = None,
        degree_inverse: int = 0,
        hyperparameter_tuning: bool = True,
        iterations_hyperparameter_tuning: int = 1500,
        n_samples: int = 2000):

    #TODO: harmonize the covariate arguement (have number_covariates, covariate_exists, covariate, also = False etc..)

    #experiment_id = mlflow.create_experiment(name="test_max_penalty",artifact_location="/Users/maherp/Desktop/Universitaet/Goettingen/5_Semester/master_thesis/mctm_pytorch/mlflow_storage/test_sim_study/")

    data_folder = "simulation_study_data/"+str(copula)+"_"+str(copula_par)+"_"+str(train_obs)+"/"

    # test data is the same for an experiment, thus we run it only once
    y_test = torch.tensor(pd.read_csv(data_folder + "grid_test.csv").values,dtype=torch.float32)
    test_log_likelihood = torch.tensor(pd.read_csv(data_folder + "test_log_likelihoods.csv").values,dtype=torch.float32).flatten()
    fig_y_test = plot_densities(y_test)

    if covariate_exists == True:
        x_test = torch.tensor(pd.read_csv(data_folder + "test_covariate.csv").values,dtype=torch.float32)
        x_test = x_test.squeeze()
        number_covariates = 1
    else:
        x_test = False
        number_covariates = 0

    # Starting the MLflow run
    mlflow.start_run(
        run_name="{}".format(seed_value),
        experiment_id=experiment_id,
        tags={"seed": seed_value,"copula": copula, "copula_par": copula_par, "train_obs": train_obs}
    )
    print("Started Run")

    # Setting the seed for reproducibility
    set_seeds(seed_value)

    # Getting the training data
    y_train = torch.tensor(pd.read_csv(data_folder+str(seed_value)+"_sample_train.csv").values,dtype=torch.float32)
    y_validate = torch.tensor(pd.read_csv(data_folder+str(seed_value)+"_sample_validate.csv").values,dtype=torch.float32)
    train_log_likelihood = torch.tensor(pd.read_csv(data_folder+str(seed_value)+"_train_log_likelihoods.csv").values,dtype=torch.float32).flatten()
    #validate_log_likelihood = torch.tensor(pd.read_csv("simulation_study_data/"+experiment_folder+str(seed_value)+"_validate_log_likelihoods.csv").values,dtype=torch.float32).flatten()

    if covariate_exists == True:
        x_train = torch.tensor(pd.read_csv(data_folder+str(seed_value)+"_train_covariate.csv").values,dtype=torch.float32)
        x_validate = torch.tensor(pd.read_csv(data_folder+str(seed_value)+"_validate_covariate.csv").values,dtype=torch.float32)
        x_train = x_train.squeeze()
        x_validate = x_validate.squeeze()
        # TODO: exchange covariate_exists to number_covariates to not have two variables
        # maybe even include in variable copula_par
    else:
        x_train = False
        x_validate = False

    if hyperparameter_tuning:

        # Running Cross validation to identify hyperparameters
        results = run_hyperparameter_tuning(y_train,
                                            y_validate,
                                            poly_span_abs, iterations_hyperparameter_tuning, spline_decorrelation,
                                        penvalueridge_list, penfirstridge_list, pensecondridge_list,
                                        lambda_penalty_params_list,
                                        learning_rate_list, patience_list, min_delta_list,
                                        degree_transformations_list, degree_decorrelation_list,
                                            x_train = x_train,
                                            x_validate = x_validate) #normalisation_layer_list

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

        # Logging the hyperparameters
    # mlflow.log_param(key="copula", value=copula)
    # mlflow.log_param(key="seed", value=seed_value)
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
    #mlflow.log_param(key="normalisation_layer", value=normalisation_layer)

    # Defining the model
    poly_range = torch.FloatTensor([[-poly_span_abs], [poly_span_abs]])
    penalty_params = torch.tensor([penvalueridge,
                                   penfirstridge,
                                   pensecondridge])

    #comes out of CV aas nan which then doe
    #normalisation_layer = None

    nf_mctm = NF_MCTM(input_min=y_train.min(0).values,
                      input_max=y_train.max(0).values,
                      polynomial_range=poly_range,
                      number_variables=y_train.size()[1],
                      spline_decorrelation=spline_decorrelation,
                      degree_transformations=int(degree_transformations),
                      degree_decorrelation=int(degree_decorrelation),
                      span_factor = span_factor,
                      span_restriction = span_restriction,
                      number_covariates=number_covariates)
                      #normalisation_layer=normalisation_layer)

    # Training the model
    loss_training_iterations, number_iterations, pen_value_ridge_final, pen_first_ridge_final, pen_second_ridge_final,\
    pen_lambda_lasso, training_time, fig_training = train(model=nf_mctm,
                                     train_data=y_train,
                                     train_covariates=x_train,
                                     penalty_params=penalty_params,
                                     lambda_penalty_params=lambda_penalty_params,
                                     iterations=iterations,
                                     learning_rate=learning_rate,
                                     patience=patience,
                                     min_delta=min_delta,
                                     verbose=False)

    # Training the inverse of the model
    fig_training_inverse = nf_mctm.l1.approximate_inverse(input=y_train,
                                                          input_covariate=x_train,
                                                          spline_inverse=spline_inverse,
                                                          degree_inverse=degree_inverse,
                                                          monotonically_increasing_inverse=monotonically_increasing_inverse,
                                                          iterations=iterations_inverse,
                                                          span_factor_inverse=span_factor_inverse,
                                                          patience=20,
                                                          global_min_loss=0.001)

    #### Training Evaluation

    sum_log_likelihood_val = nf_mctm.log_likelihood(y_validate, x_validate).detach().numpy().sum()

    fig_y_train = plot_densities(y_train, x_lim=[y_train.min(),y_train.max()], y_lim=[y_train.min(),y_train.max()])

    fig_splines_transformation_layer_1 = plot_splines(layer= nf_mctm.l1, y_train=y_train, covariate_exists=covariate_exists)
    fig_splines_decorrelation_layer_2 = plot_splines(layer= nf_mctm.l2, covariate_exists=covariate_exists)
    fig_splines_decorrelation_layer_4 = plot_splines(layer= nf_mctm.l4, covariate_exists=covariate_exists)
    fig_splines_decorrelation_layer_6 = plot_splines(layer= nf_mctm.l6, covariate_exists=covariate_exists)

    # Evaluate latent space of the model in training set
    z_train = nf_mctm.latent_space_representation(y_train, x_train).detach().numpy()
    res_normal_train, res_pval_train, z_mean_train, z_cov_train, p_train = evaluate_latent_space(z_train)
    fig_z_train = plot_densities(z_train, x_train)

    predicted_train_log_likelihood = nf_mctm.log_likelihood(y_train, x_train)
    kl_divergence_nf_mctm_train_vec = kl_divergence(target_log_likelihood=train_log_likelihood,
                                                predicted_log_likelihood=predicted_train_log_likelihood,
                                                mean=False)
    kl_divergence_nf_mctm_train = torch.mean(kl_divergence_nf_mctm_train_vec).item()
    fig_kl_divergence_nf_mctm_train = plot_kl_divergence_scatter(y_train, kl_divergence_nf_mctm_train_vec)

    # estimate true model on training data
    train_log_likelihood_estimated_true_model = torch.tensor(pd.read_csv(data_folder + str(seed_value) + "_est_train_log_likelihoods.csv").values,dtype=torch.float32).flatten()
    kl_divergence_true_model_train_vec = kl_divergence(target_log_likelihood=train_log_likelihood,
                                                   predicted_log_likelihood=train_log_likelihood_estimated_true_model,
                                                   mean=False)
    kl_divergence_true_model_train = torch.mean(kl_divergence_true_model_train_vec).item()
    fig_kl_divergence_true_model_train = plot_kl_divergence_scatter(y_train, kl_divergence_true_model_train_vec)

    # estimate the Multivariate Normal Distribution as Model
    mean_mvn_model = y_train.mean(0) #0 to do mean across dim 0 not globally
    cov_mvn_model = y_train.T.cov()
    mvn_model = MultivariateNormal(loc=mean_mvn_model, covariance_matrix=cov_mvn_model)

    train_log_likelihood_mvn_model = mvn_model.log_prob(y_train)
    kl_divergence_mvn_model_train_vec = kl_divergence(target_log_likelihood=train_log_likelihood,
                                                  predicted_log_likelihood=train_log_likelihood_mvn_model,
                                                  mean=False)
    kl_divergence_mvn_model_train = torch.mean(kl_divergence_mvn_model_train_vec).item()
    fig_kl_divergence_mvn_model_train = plot_kl_divergence_scatter(y_train, kl_divergence_mvn_model_train_vec)

    #### Test Evaluation
    # Evaluate latent space of the model in test set
    z_test = nf_mctm.latent_space_representation(y_test, x_test).detach().numpy()
    res_normal_test, res_pval_test, z_mean_test, z_cov_test, p_test = evaluate_latent_space(z_test)
    fig_z_test = plot_densities(z_test)

    predicted_test_log_likelihood = nf_mctm.log_likelihood(y_test, x_test)
    kl_divergence_nf_mctm_test_vec = kl_divergence(target_log_likelihood=test_log_likelihood,
                                                predicted_log_likelihood=predicted_test_log_likelihood,
                                                mean=False)
    kl_divergence_nf_mctm_test = torch.mean(kl_divergence_nf_mctm_test_vec).item()
    fig_kl_divergence_nf_mctm_test = plot_kl_divergence_scatter(y_test, kl_divergence_nf_mctm_test_vec)

    # estimated true model on test data
    test_log_likelihood_estimated_true_model = torch.tensor(pd.read_csv(data_folder + str(seed_value) + "_est_test_log_likelihoods.csv").values,dtype=torch.float32).flatten()
    kl_divergence_true_model_test_vec = kl_divergence(target_log_likelihood=test_log_likelihood,
                                                   predicted_log_likelihood=test_log_likelihood_estimated_true_model,
                                                   mean=False)
    kl_divergence_true_model_test = torch.mean(kl_divergence_true_model_test_vec).item()
    fig_kl_divergence_true_model_test = plot_kl_divergence_scatter(y_test, kl_divergence_true_model_test_vec)

    # mvn Model on test data
    test_log_likelihood_mvn_model = mvn_model.log_prob(y_test)
    kl_divergence_mvn_model_test_vec = kl_divergence(target_log_likelihood=test_log_likelihood,
                                                  predicted_log_likelihood=test_log_likelihood_mvn_model,
                                                  mean=False)
    kl_divergence_mvn_model_test = torch.mean(kl_divergence_mvn_model_test_vec).item()
    fig_kl_divergence_mvn_model_test = plot_kl_divergence_scatter(y_test, kl_divergence_mvn_model_test_vec)

    #### Generate a sample from the trained model
    # TODO: discuss this, I sample from model with x_train as values (sensible buy maybe moroe specific values better?)
    if x_train is not False:
        x_sample = x_train[x_train.multinomial(num_samples=n_samples, replacement=True)]
    else:
        x_sample = False
    y_sampled = nf_mctm.sample(n_samples=n_samples, covariate=x_sample)
    y_sampled = y_sampled.detach().numpy()
    fig_y_sampled = plot_densities(y_sampled,
                                   x_lim=[y_train.min(), y_train.max()],
                                   y_lim=[y_train.min(), y_train.max()])

    #### Log Training Artifacts
    model_info = mlflow.pytorch.log_model(nf_mctm, "nf_mctm_model")

    log_mlflow_plot(fig_training,'plot_training.png')
    log_mlflow_plot(fig_training_inverse,'plot_training_inverse.png')

    #fig_training.savefig('plot_training.png')
    #mlflow.log_artifact("./plot_training.png")
    #fig_training_inverse.savefig('plot_training_inverse.png')
    #mlflow.log_artifact("./plot_training_inverse.png")
    mlflow.log_metric("number_iterations", number_iterations)
    mlflow.log_metric("training_time", training_time)
    mlflow.log_metric("sum_log_likelihood_validation", sum_log_likelihood_val)

    mlflow.log_metric("pen_value_ridge_final", pen_value_ridge_final)
    mlflow.log_metric("pen_first_ridge_final", pen_first_ridge_final)
    mlflow.log_metric("pen_second_ridge_final", pen_second_ridge_final)

    np.save("loss_training_iterations.npy", loss_training_iterations)
    mlflow.log_artifact("./loss_training_iterations.npy")

    log_mlflow_plot(fig_splines_transformation_layer_1,'plot_splines_transformation_layer_1.png')
    log_mlflow_plot(fig_splines_decorrelation_layer_2,'plot_splines_decorrelation_layer_2.png')
    log_mlflow_plot(fig_splines_decorrelation_layer_4,'plot_splines_decorrelation_layer_4.png')
    log_mlflow_plot(fig_splines_decorrelation_layer_6,'plot_splines_decorrelation_layer_6.png')

    #fig_splines_transformation_layer_1.savefig('plot_splines_transformation_layer_1.png')
    #mlflow.log_artifact("./plot_splines_transformation_layer_1.png")
    #fig_splines_decorrelation_layer_2.savefig('plot_splines_decorrelation_layer_2.png')
    #mlflow.log_artifact("./plot_splines_decorrelation_layer_2.png")
    #fig_splines_decorrelation_layer_4.savefig('plot_splines_decorrelation_layer_4.png')
    #mlflow.log_artifact("./plot_splines_decorrelation_layer_4.png")
    #fig_splines_decorrelation_layer_6.savefig('plot_splines_decorrelation_layer_6.png')
    #mlflow.log_artifact("./plot_splines_decorrelation_layer_6.png")

    #if hyperparameter_tuning:
        #results.to_csv("hyperparameter_tuning_results.csv")
        #mlflow.log_artifact("./hyperparameter_tuning_results.csv")

        #results_summary.to_csv("hyperparameter_tuning_results_summary.csv")
        #mlflow.log_artifact("./hyperparameter_tuning_results_summary.csv")

    #### Log Train Data Metrics and Artifacts
    log_mlflow_plot(fig_y_train,'plot_data_train.png')
    #fig_y_train.savefig('plot_data_train.png')
    #mlflow.log_artifact("./plot_data_train.png")

    mlflow.log_metric("mv_normality_result_train", res_normal_train)
    mlflow.log_metric("mv_normality_pval_train", res_pval_train)

    np.save("z_mean_train.npy", z_mean_train)
    mlflow.log_artifact("./z_mean_train.npy")

    np.save("z_cov_train.npy", z_cov_train)
    mlflow.log_artifact("./z_cov_train.npy")

    np.save("uv_normality_pvals_train.npy", p_train)
    mlflow.log_artifact("./uv_normality_pvals_train.npy")

    log_mlflow_plot(fig_z_train,'plot_latent_space_train.png')
    #fig_z_train.savefig('plot_latent_space_train.png')
    #mlflow.log_artifact("./plot_latent_space_train.png")

    mlflow.log_metric("kl_divergence_nf_mctm_train", kl_divergence_nf_mctm_train)
    mlflow.log_metric("kl_divergence_true_model_train", kl_divergence_true_model_train)
    mlflow.log_metric("kl_divergence_mvn_model_train", kl_divergence_mvn_model_train)

    log_mlflow_plot(fig_kl_divergence_nf_mctm_train,'plot_kl_divergence_nf_mctm_train.png')
    #fig_kl_divergence_nf_mctm_train.savefig('plot_kl_divergence_nf_mctm_train.png')
    #mlflow.log_artifact("./plot_kl_divergence_nf_mctm_train.png")

    log_mlflow_plot(fig_kl_divergence_true_model_train,'plot_kl_divergence_true_model_train.png')
    #fig_kl_divergence_true_model_train.savefig('plot_kl_divergence_true_model_train.png')
    #mlflow.log_artifact("./plot_kl_divergence_true_model_train.png")

    log_mlflow_plot(fig_kl_divergence_mvn_model_train,'plot_kl_divergence_mvn_model_train.png')
    #fig_kl_divergence_mvn_model_train.savefig('plot_kl_divergence_mvn_model_train.png')
    #mlflow.log_artifact("./plot_kl_divergence_mvn_model_train.png")

    #### Log Test Data Metrics and Artifacts
    log_mlflow_plot(fig_y_test,'plot_data_test.png')
    #fig_y_test.savefig('plot_data_test.png')
    #mlflow.log_artifact("./plot_data_test.png")

    mlflow.log_metric("mv_normality_result_test", res_normal_test)
    mlflow.log_metric("mv_normality_pval_test", res_pval_test)

    np.save("z_mean_test.npy", z_mean_test)
    mlflow.log_artifact("./z_mean_test.npy")

    np.save("z_cov_test.npy", z_cov_test)
    mlflow.log_artifact("./z_cov_test.npy")

    np.save("uv_normality_pvals_test.npy", p_test)
    mlflow.log_artifact("./uv_normality_pvals_test.npy")

    log_mlflow_plot(fig_z_test,'plot_latent_space_test.png')
    #fig_z_test.savefig('plot_latent_space_test.png')
    #mlflow.log_artifact("./plot_latent_space_test.png")

    mlflow.log_metric("kl_divergence_nf_mctm_test", kl_divergence_nf_mctm_test)
    mlflow.log_metric("kl_divergence_true_model_test", kl_divergence_true_model_test)
    mlflow.log_metric("kl_divergence_mvn_model_test", kl_divergence_mvn_model_test)

    log_mlflow_plot(fig_kl_divergence_nf_mctm_test,'plot_kl_divergence_nf_mctm_test.png')
    #fig_kl_divergence_nf_mctm_test.savefig('plot_kl_divergence_nf_mctm_test.png')
    #mlflow.log_artifact("./plot_kl_divergence_nf_mctm_test.png")

    log_mlflow_plot(fig_kl_divergence_true_model_test,'plot_kl_divergence_true_model_test.png')
    #fig_kl_divergence_true_model_test.savefig('plot_kl_divergence_true_model_test.png')
    #mlflow.log_artifact("./plot_kl_divergence_true_model_test.png")

    log_mlflow_plot(fig_kl_divergence_mvn_model_test,'plot_kl_divergence_mvn_model_test.png')
    #fig_kl_divergence_mvn_model_test.savefig('plot_kl_divergence_mvn_model_test.png')
    #mlflow.log_artifact("./plot_kl_divergence_mvn_model_test.png")

    #### Log Sampling Aritfacts
    np.save("synthetically_sampled_data.npy", y_sampled)
    mlflow.log_artifact("./synthetically_sampled_data.npy")

    log_mlflow_plot(fig_y_sampled,'plot_synthetically_sampled_data.png')
    #fig_y_sampled.savefig('plot_synthetically_sampled_data.png')
    #mlflow.log_artifact("./plot_synthetically_sampled_data.png")

    #create table of all coefficients of the bsplines
    bspline_table = pd.DataFrame()
    bspline_table["bspline_1"] = nf_mctm.l2.params.detach().numpy().flatten()
    bspline_table["bspline_2"] = nf_mctm.l4.params.detach().numpy().flatten()
    bspline_table["bspline_3"] = nf_mctm.l6.params.detach().numpy().flatten()
    bspline_table.to_csv("bspline_table.csv")
    mlflow.log_artifact("./bspline_table.csv")

    # End the run
    print("Finished Run")
    mlflow.end_run()

if __name__ == '__main__':

    run_simulation_study(
        experiment_id = 464499768340700910,
        copula = "3d_joe",
        copula_par = "3",
        train_obs = 2000,
        covariate_exists = False,
        # Setting Hyperparameter Values
        seed_value=1,
        penvalueridge_list=[0],
        penfirstridge_list=[0],
        pensecondridge_list=[0],
        poly_span_abs=15,
        spline_decorrelation="bspline",
        spline_inverse="bernstein",
        span_factor=0.1,
        span_factor_inverse=0.2,
        span_restriction="reluler",
        iterations=2000,
        iterations_hyperparameter_tuning=5000,
        iterations_inverse=1,
        learning_rate_list=[1.], #TODO: irrelevant as we use line search for the learning rate
        patience_list=[10],
        min_delta_list=[1e-8],
        degree_transformations_list=[15],
        degree_decorrelation_list=[40],
        lambda_penalty_params_list=[False],#[torch.tensor([[0,1],[1,0]])],
        #normalisation_layer_list=[None],
        degree_inverse=40,
        monotonically_increasing_inverse=True,
        hyperparameter_tuning=False,
        n_samples=2000)
    #TODO: stop the plots all from showing plots

