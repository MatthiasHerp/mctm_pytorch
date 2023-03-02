import torch
import pathlib
from python_nf_mctm.simulation_study_helpers import *
from python_nf_mctm.nf_mctm import *
import mlflow
import torch

# this file load
if __name__ == "__main__":
    experiment_metadata = mlflow.search_experiments()

    for i in range(len(experiment_metadata)):
        experiment_metadata[i] = dict(experiment_metadata[i])

    # get all names of experiments
    experiment_names = [experiment_metadata[i]['name'] for i in range(len(experiment_metadata))]

    experiment_names = ["joe_3_500","joe_3_1000","joe_3_2000"]#["joe_covariate_2000"]#["t_3_500","t_3_1000","t_3_2000"]
    y_train = torch.tensor([[-5, -5], [5, 5]])

    for experiment in experiment_names:
        # get all runs of experiment

        runs = mlflow.search_runs(experiment_names=[experiment])

        #index_pos = runs[runs["run_id"] == "c8fec629733c48fd854b9a82bb767f63"].index.item()
        #index_pos = 23

        for run_artifact_uri in runs["artifact_uri"]:#[index_pos:]:
            run_artifact_uri = run_artifact_uri[run_artifact_uri.find('mlruns'):]

            if not pathlib.Path(run_artifact_uri + "/plot_splines_transformation_layer_1_updated.png").is_file():
                #load model
                model = mlflow.pytorch.load_model(run_artifact_uri+"/nf_mctm_model")

                #rerun inverse with false input data, we simp
                model.l1.approximate_inverse(input=y_train,
                                             spline_inverse="bspline",
                                             monotonically_increasing_inverse=False,
                                             degree_inverse=150,
                                             span_factor_inverse=0.2)

                fig_splines = plot_splines(model.l1, y_train=y_train)

                fig_splines.savefig(pathlib.Path(run_artifact_uri + "/plot_splines_transformation_layer_1_updated.png"))

                n_samples = 2000
                x_sample = False
                y_sampled = model.sample(n_samples=n_samples, covariate=x_sample)
                y_sampled = y_sampled.detach().numpy()
                fig_y_sampled = plot_densities(y_sampled,
                                               covariate=x_sample)

                fig_y_sampled.savefig(pathlib.Path(run_artifact_uri + "/plot_synthetically_sampled_data_updated.png"))

                #save model
                #mlflow.pytorch.save_model(run_artifact_uri+"/nf_mctm_model")
                #mlflow.pytorch.save_model(model, path=pathlib.Path(run_artifact_uri[90:] + "/nf_mctm_model_updated"))

                #mlflow.pytorch.save_model(model, path=pathlib.Path(run_artifact_uri + "/nf_mctm_model_updated"))

                mlflow.pytorch.save_model(model, path=pathlib.Path(run_artifact_uri[7:] + "/nf_mctm_model_updated"))
            else:
                print("Already done with run_artifact_uri: " + run_artifact_uri)

        print("Done with experiment: " + experiment)


    print("Done!")

