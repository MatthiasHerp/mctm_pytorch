import numpy as np
import pandas as pd
import torch

from simulation_study_helpers import *
from training_helpers import *
from nf_mctm import *
import mlflow

def run_simulation_study(
        copula,
        # Setting Hyperparameter Values
        seed_value,
        penvalueridge,
        penfirstridge,
        pensecondridge,
        poly_span_abs,
        spline_decorrelation,
        iterations,
        learning_rate,
        patience,
        min_delta,
        degree_transformations,
        degree_decorrelation,
        normalisation_layer):

    #experiment_id = mlflow.create_experiment(name="test_sim_study",artifact_location="/Users/maherp/Desktop/Universitaet/Goettingen/5_Semester/master_thesis/mctm_pytorch/mlflow_storage/test_sim_study/")
    experiment_id = 1

    # test data is the same for an experiment
    y_test = torch.tensor(pd.read_csv("simulation_study_data/"+str(copula)+"_3_2000/" + "grid_test.csv").values,dtype=torch.float32)
    test_log_likelihood = torch.tensor(pd.read_csv("simulation_study_data/"+str(copula)+"_3_2000/" + "test_log_likelihoods.csv").values,dtype=torch.float32).flatten()

    # Starting the MLflow run
    mlflow.start_run(
        run_name="{}".format(seed_value),
        experiment_id=experiment_id,
        tags={"seed": seed_value,"copula": copula}
    )
    print("Started Run")

    # Logging the hyperparameters
    #mlflow.log_param(key="copula", value=copula)
    #mlflow.log_param(key="seed", value=seed_value)
    mlflow.log_param(key="pen_value_ridge", value=penvalueridge)
    mlflow.log_param(key="pen_first_ridge", value=penfirstridge)
    mlflow.log_param(key="pen_second_ridge", value=pensecondridge)
    mlflow.log_param(key="poly_span_abs", value=poly_span_abs)
    mlflow.log_param(key="spline_decorrelation", value=spline_decorrelation)
    mlflow.log_param(key="iterations", value=iterations)
    mlflow.log_param(key="learning_rate", value=learning_rate)
    mlflow.log_param(key="patience", value=patience)
    mlflow.log_param(key="min_delta", value=min_delta)
    mlflow.log_param(key="degree_transformations", value=degree_transformations)
    mlflow.log_param(key="degree_decorrelation", value=degree_decorrelation)
    mlflow.log_param(key="normalisation_layer", value=normalisation_layer)

    # Setting the seed for reproducibility
    set_seeds(seed_value)

    # Getting the training data
    y_train = torch.tensor(pd.read_csv("simulation_study_data/"+str(copula)+"_3_2000/"+str(seed_value)+"_sample_train.csv").values,dtype=torch.float32)
    train_log_likelihood = torch.tensor(pd.read_csv("simulation_study_data/"+str(copula)+"_3_2000/"+str(seed_value)+"_train_log_likelihoods.csv").values,dtype=torch.float32).flatten()

    # Defining the model
    poly_range = torch.FloatTensor([[-poly_span_abs], [poly_span_abs]])
    penalty_params = torch.tensor([penvalueridge,
                                   penfirstridge,
                                   pensecondridge])
    nf_mctm = NF_MCTM(input_min=y_train.min(0).values,
                      input_max=y_train.max(0).values,
                      polynomial_range=poly_range,
                      number_variables=y_train.size()[1],
                      spline_decorrelation=spline_decorrelation,
                      degree_transformations=degree_transformations,
                      degree_decorrelation=degree_decorrelation,
                      normalisation_layer=normalisation_layer)

    # Training the model
    loss_training_iterations, fig_training = train(model=nf_mctm,
                                     train_data=y_train,
                                     penalty_params=penalty_params,
                                     iterations=iterations,
                                     learning_rate=learning_rate,
                                     patience=patience,
                                     min_delta=min_delta,
                                     verbose=False)

    #### Training Evaluation
    # Evaluate latent space of the model in training set
    z_train = nf_mctm.forward(y_train, train=False).detach().numpy()
    res_normal_train, res_pval_train, z_mean_train, z_cov_train, p_train = evaluate_latent_space(z_train)
    fig_train = plot_latent_space(z_train)

    predicted_train_log_likelihood = nf_mctm.log_likelihood(y_train)
    kl_divergence_nf_mctm_train = kl_divergence(target_log_likelihood=train_log_likelihood,
                                                predicted_log_likelihood=predicted_train_log_likelihood)

    # estimate true model on training data
    train_log_likelihood_estimated_true_model = torch.tensor(pd.read_csv("simulation_study_data/"+str(copula)+"_3_2000/" + str(seed_value) + "_est_train_log_likelihoods.csv").values,dtype=torch.float32).flatten()
    kl_divergence_true_model_train = kl_divergence(target_log_likelihood=train_log_likelihood,
                                                   predicted_log_likelihood=train_log_likelihood_estimated_true_model)

    #### Test Evaluation
    # Evaluate latent space of the model in test set
    z_test = nf_mctm.forward(y_test, train=False).detach().numpy()
    res_normal_test, res_pval_test, z_mean_test, z_cov_test, p_test = evaluate_latent_space(z_test)
    fig_test = plot_latent_space(z_test)

    predicted_test_log_likelihood = nf_mctm.log_likelihood(y_test)
    kl_divergence_nf_mctm_test = kl_divergence(target_log_likelihood=test_log_likelihood,
                                                predicted_log_likelihood=predicted_test_log_likelihood)

    # estimate true model on test data
    test_log_likelihood_estimated_true_model = torch.tensor(pd.read_csv("simulation_study_data/"+str(copula)+"_3_2000/" + str(seed_value) + "_est_test_log_likelihoods.csv").values,dtype=torch.float32).flatten()
    kl_divergence_true_model_test = kl_divergence(target_log_likelihood=test_log_likelihood,
                                                   predicted_log_likelihood=test_log_likelihood_estimated_true_model)

    #### Log Training Artifacts
    model_info = mlflow.pytorch.log_model(nf_mctm, "nf_mctm_model")
    fig_training.savefig('plot_training.png')
    mlflow.log_artifact("./plot_training.png")

    np.save("loss_training_iterations.npy", loss_training_iterations)
    mlflow.log_artifact("./loss_training_iterations.npy")

    #### Log Train Data Metrics and Artifacts
    mlflow.log_metric("mv_normality_result_train", res_normal_train)
    mlflow.log_metric("mv_normality_pval_train", res_pval_train)

    np.save("z_mean_train.npy", z_mean_train)
    mlflow.log_artifact("./z_mean_train.npy")

    np.save("z_cov_train.npy", z_cov_train)
    mlflow.log_artifact("./z_cov_train.npy")

    np.save("uv_normality_pvals_train.npy", p_train)
    mlflow.log_artifact("./uv_normality_pvals_train.npy")

    fig_train.savefig('plot_latent_space_train.png')
    mlflow.log_artifact("./plot_latent_space_train.png")

    mlflow.log_metric("kl_divergence_nf_mctm_train", kl_divergence_nf_mctm_train)
    mlflow.log_metric("kl_divergence_true_model_train", kl_divergence_true_model_train)

    #### Log Test Data Metrics and Artifacts
    mlflow.log_metric("mv_normality_result_test", res_normal_test)
    mlflow.log_metric("mv_normality_pval_train", res_pval_test)

    np.save("z_mean_test.npy", z_mean_test)
    mlflow.log_artifact("./z_mean_test.npy")

    np.save("z_cov_test.npy", z_cov_test)
    mlflow.log_artifact("./z_cov_test.npy")

    np.save("uv_normality_pvals_test.npy", p_test)
    mlflow.log_artifact("./uv_normality_pvals_test.npy")

    fig_test.savefig('plot_latent_space_test.png')
    mlflow.log_artifact("./plot_latent_space_test.png")

    mlflow.log_metric("kl_divergence_nf_mctm_test", kl_divergence_nf_mctm_test)
    mlflow.log_metric("kl_divergence_true_model_test", kl_divergence_true_model_test)

    # End the run
    print("Finished Run")
    mlflow.end_run()

if __name__ == '__main__':

    run_simulation_study(
        copula="joe",
        # Setting Hyperparameter Values
        seed_value=1,
        penvalueridge=0,
        penfirstridge=0,
        pensecondridge=0,
        poly_span_abs=5,
        spline_decorrelation="bspline",
        iterations=1000,
        learning_rate=0.5,
        patience=10,
        min_delta=1e-8,
        degree_transformations=10,
        degree_decorrelation=10,
        normalisation_layer=None)

