import mlflow
from run_simulation_study import run_simulation_study

if __name__ == "__main__":

    #iopwwwwwwwwmlflow.create_experiment(name="joe_3_2000")
    experiment = mlflow.get_experiment_by_name("joe_3_2000")



    for seed_num in range(1,50):

        run_simulation_study(
                experiment_id = experiment.experiment_id,
                copula = "joe",
                copula_par = 3,
                train_obs = 2000,
                # Setting Hyperparameter Values
                seed_value=seed_num,
                penvalueridge_list=[0],
                penfirstridge_list=[0],
                pensecondridge_list=[1],
                poly_span_abs=15,
                spline_decorrelation="bspline",
                spline_inverse="bernstein",
                span_factor=0.1,
                span_factor_inverse=0.2,
                span_restriction="reluler",
                iterations=10,
                iterations_hyperparameter_tuning=5000,
                iterations_inverse=1,
                learning_rate_list=[1.],
                patience_list=[10],
                min_delta_list=[1e-8],
                degree_transformations_list=[15],
                degree_decorrelation_list=[40],
                #normalisation_layer_list=[None],
                degree_inverse=40,
                monotonically_increasing_inverse=True,
                hyperparameter_tuning=True,
                n_samples=2000)