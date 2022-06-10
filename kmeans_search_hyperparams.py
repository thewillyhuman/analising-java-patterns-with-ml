import argparse
import os
import sys
from subprocess import check_call

from models.utils import Params

PYTHON = sys.executable

parser = argparse.ArgumentParser()
parser.add_argument('--parent-dir', required=True, help="Directory containing params.json")
parser.add_argument('--data-dir', required=True, help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} kmeans_fit.py --model-dir {model_dir} --data-dir {data_dir}".format(python=PYTHON,
            model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Perform hypersearch over one parameter
    n_clusters_list = [2, 3, 4, 5, 6, 7, 8, 9]

    for n_clusters in n_clusters_list:
        # Modify the relevant parameter in params
        params.n_clusters = n_clusters

        # Launch job (name has to be unique)
        job_name = "_n_clusters_{}".format(n_clusters)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)