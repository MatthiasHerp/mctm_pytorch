import mlflow
from run_simulation_study import run_simulation_study
import torch

if __name__ == "__main__":

    #mlflow.create_experiment(name="t_3_2000")
    experiment = mlflow.get_experiment_by_name("t_3_2000_6_decorr")

    #source: https://discuss.pytorch.org/t/how-to-use-multi-cpu-or-muti-cpu-core-to-train/147124
    torch.set_num_interop_threads(50)  # Inter-op parallelism
    torch.set_num_threads(50)  # Intra-op parallelism

    for seed in range(1,51):

        run_simulation_study(
                experiment_id = experiment.experiment_id,
                copula = "t",
                copula_par = 3,
                covariate_exists=False,
                train_obs = 2000,
                # Setting Hyperparameter Values
                seed_value=seed,
                penvalueridge_list=[0],
                penfirstridge_list=[0],
                pensecondridge_list=[0],
                poly_span_abs=15,
                spline_decorrelation="bspline",
                spline_inverse="bspline",
                span_factor=0.1,
                span_factor_inverse=0.2,
                span_restriction="reluler",
                iterations=10000,
                iterations_hyperparameter_tuning=5000,
                iterations_inverse=5000,
                learning_rate_list=[1.],
                patience_list=[10],
                min_delta_list=[1e-8],
                degree_transformations_list=[30],
                degree_decorrelation_list=[40],
                lambda_penalty_params_list=[False],
                #normalisation_layer_list=[None],
                degree_inverse=150,
                monotonically_increasing_inverse=True,
                hyperparameter_tuning=True,
                n_samples=2000,
                num_decorr_layers=6)