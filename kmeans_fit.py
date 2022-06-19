"""
Creates and fits a KMeans model (scikit-learn KMeans) over the given dataset.

Run: python kmeans_fit.py \
        --data-dir <data-folder> \
        --model-dir <dir-to-store-model> \
        [--compute-elbow] \
        [--max-n-features <number_of_max_features>]
"""
import argparse
import logging
import os

from joblib import dump
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from tabulate import tabulate

from build_dataset import normalize_datatypes
from models.kmeans import find_elbow, build_model, fit_and_evaluate, \
    explore_distributions_and_centroids
from models.utils import Params, set_logger, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--model-dir', type=str, required=True)
parser.add_argument('--compute-elbow', action='store_true')
parser.add_argument('--max-n-features', type=int, default=-1)

if __name__ == '__main__':

    # Load the data for the experiment
    args = parser.parse_args()
    dataset = Dataset(data_dir=args.data_dir)
    model_path = os.path.join(args.model_dir, 'model.joblib')

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Load the parameters from the experiment params.json file in model_dir
    json_path = os.path.join(args.model_dir, 'params.json')
    if os.path.isfile(json_path):
        params = Params(json_path)
        logging.info(f"Parameters file found at [{json_path}].")
    else:
        # Write parameters in json file
        params = None
        logging.info("Parameters file not found.")

    # Load the dataset
    x, y = dataset.x, dataset.y
    logging.info(f"Dataset loaded. Shape: {x.shape}, {y.shape}.")

    # OneHot encoding
    x, y, features = normalize_datatypes(x, y)
    logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

    if args.max_n_features != -1:
        logging.info(f"Selecting {args.max_n_features} features.")
        selector = SelectKBest(mutual_info_regression, k=args.max_n_features)
        selector.fit(x, y)
        print(x.columns[selector.get_support()].values.tolist())
        x = x[x.columns[selector.get_support()].values.tolist()]
        features = x.columns.values.tolist()

    if args.compute_elbow:
        logging.info(f"Option --compute-elbow present. Overriding params.json elbow.")
        number_of_clusters = find_elbow(x, 2, 15)
        params.n_clusters = number_of_clusters
        params.save(json_path)
    else:
        logging.info(f"Option --compute-elbow not present. Using params.json elbow "
                     f"[{params.n_clusters}].")

    # Create the model
    logging.info("Creating model.")
    kmeans_model = build_model(params=params)

    # Train the model
    logging.info("Starting training.")
    model, metrics_df = fit_and_evaluate(model=kmeans_model, x=x, y=y, target_name=dataset.target)
    logging.info("Training finished.")

    logging.info(
        "Evaluation: \n" + tabulate(metrics_df, headers='keys', tablefmt='tsv', floatfmt=".2f"))

    params.inertia = model.inertia_
    params.n_iter = model.n_iter_
    params.n_features_in = model.n_features_in_

    # Write parameters in json file.
    json_path = os.path.join(args.model_dir, 'metrics_eval_best_weights.json')
    params.save(json_path)

    # Write model in pickle file.
    model_path = os.path.join(args.model_dir, 'model.joblib')
    dump(model, model_path)

    logging.info("Data saved.")

    logging.info("")
    logging.info("")
    logging.info("")
    explore_distributions_and_centroids(model, x, y, dataset.numeric_features, dataset.target)
    logging.info("")
    logging.info("")
    logging.info("")
    logging.info(
        "Centroids: \n" + tabulate(metrics_df.T, headers='keys', tablefmt='tsv', floatfmt=".2f"))
    logging.info("All saved and finished.")
