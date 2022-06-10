import argparse
import logging
import os

from build_dataset import download_data, normalize_datatypes, scale_data_to_range_0_1
from models.models import build_elastic_log_reg_model, train_and_evaluate_log_reg, save_log_reg_coefs_to_excel, \
    build_kmeans_model, fit_and_evaluate_kmeans, search_kmeans_elbow
from models.utils import Params, set_logger
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True, help="Directory containing query and features")
parser.add_argument('--model-dir', type=str, required=True, help="Directory containing params.json")
parser.add_argument('--compute-elbow', action='store_true', help="Whether to compute the number of clusters or not")
parser.add_argument('--database-name', type=str, default='patternminingV2')


if __name__ == '__main__':

    # Load the data for the experiment
    args = parser.parse_args()
    query_path = os.path.join(args.data_dir, 'query.sql')
    features_path = os.path.join(args.data_dir, 'features.txt')
    target_path = os.path.join(args.data_dir, 'target.txt')
    percentage_features_path = os.path.join(args.data_dir, 'percentage_features.txt')
    assert os.path.isfile(query_path), "No json configuration file found at {}".format(query_path)
    assert os.path.isfile(features_path), "No json configuration file found at {}".format(features_path)
    assert os.path.isfile(percentage_features_path), "No json configuration file found at {}".format(percentage_features_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Load the parameters from the experiment params.json file in model_dir
    json_path = os.path.join(args.model_dir, 'params.json')
    if os.path.isfile(json_path):
        params = Params(json_path)
        logging.info(f"Parameters file found {params}.")
    else:
        # Write parameters in json file
        params = None
        logging.info("Parameters file not found.")

    # Load the dataset
    logging.info("Downloading dataset...")
    query = open(query_path, mode='r').read()
    features = [feature.rstrip() for feature in open(features_path, mode='r').readlines()]
    target = open(target_path, mode='r').read()
    percentage_features = [feature.rstrip() for feature in open(percentage_features_path, mode='r').readlines()]
    x, y = download_data(query=query, database_name=args.database_name, features=features, target=target)
    logging.info("Downloaded dataset. Features shape {}. Target shape {}.".format(x.shape, y.shape))

    # OneHot encoding
    x, y, features = normalize_datatypes(x, y)
    logging.info("Features normalized. Features shape {}. Target shape {}.".format(x.shape, y.shape))

    # Scale features to range 0 1
    x = scale_data_to_range_0_1(x, features, percentage_features)
    logging.info("Features scaled. Features shape {}. Target shape {}.".format(x.shape, y.shape))

    if args.compute_elbow:
        # Find best number of clusters (elbow)
        number_of_clusters = search_kmeans_elbow(x, 2, 15)
        params.n_clusters = number_of_clusters
        params.save(json_path)

    # Create the model
    logging.info("Creating model for params [{}]...".format(params))
    kmeans_model = build_kmeans_model(params=params)

    # Train the model
    logging.info("Starting training")
    model, metrics_df = fit_and_evaluate_kmeans(kmeans=kmeans_model, x=x, y=y, target_name=target)
    logging.info("Training finished.")

    logging.info("Evaluation: \n" + tabulate(metrics_df, headers = 'keys', tablefmt = 'psql'))

    params.inertia = model.inertia_
    params.n_iter = model.n_iter_
    params.n_features_in = model.n_features_in_

    # Write parameters in json file
    json_path = os.path.join(args.model_dir, 'metrics_eval_best_weights.json')
    params.save(json_path)
    logging.info("Data saved.")

    for row in model.cluster_centers_:
        cluster_rule = "["
        column_index = 0;
        for column in row:
            cluster_rule = cluster_rule + f"{features[column_index]}={column:.8f}"
            if column_index != len(row) - 1:
                cluster_rule = cluster_rule + " ∧ "
            column_index = column_index + 1
        cluster_rule = cluster_rule + "]"
        logging.info(cluster_rule)

    #logging.info(tabulate(model.cluster_centers_, headers=features, tablefmt='psql'))

    logging.info("All saved and finished.")