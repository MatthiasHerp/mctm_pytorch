"""CLI script to fix artifact paths when `mlruns` directory is moved.
This could happen if you rename a parent directory when running locally, or if
you download an `mlruns/` folder produced on a different machine/file system.
Args:
  path: Path to mlruns folder containing experiments.
"""

import yaml
from pathlib import Path
import argparse


def rewrite_artifact_path(metadata_file, pwd, artifact_path_key):
    with open(metadata_file, "r") as f:
        y = yaml.safe_load(f)
        y[artifact_path_key] = f"file://{pwd}"

    with open(metadata_file, "w") as f:
        #print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("path", help="Path to root of `mlruns` folder.")

    #args = parser.parse_args()

    #absolute_path = Path(args.path).resolve()

    absolute_path = "/Users/maherp/Desktop/Universitaet/Goettingen/5_Semester/master_thesis/mctm_pytorch/mlruns"
    absolute_path = Path(absolute_path).resolve()

    #print(absolute_path)
    for experiment_folder in absolute_path.iterdir():

        if ".DS_Store" not in str(experiment_folder):
            #print(experiment_folder)

            metadata_file = experiment_folder / "meta.yaml"

            # Fix experiment metadata
            if metadata_file.exists():
                rewrite_artifact_path(metadata_file, experiment_folder, artifact_path_key='artifact_location')
            for run_folder in experiment_folder.iterdir():
                metadata_file = run_folder / "meta.yaml"
                #print(run_folder)

                # Fix run metadata
                if metadata_file.exists():
                    rewrite_artifact_path(metadata_file, run_folder / "artifacts", artifact_path_key='artifact_uri')

    print("Done!")