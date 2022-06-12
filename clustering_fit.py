import argparse
import logging
import os

import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, OPTICS

from build_dataset import load_dataset_from_csv, normalize_datatypes, scale_data_to_range_0_1
from models.utils import Params, set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True, help="Directory containing query and features")
parser.add_argument('--model-dir', type=str, required=True, help="Directory containing params.json")


def fit_kmeans(x: pd.DataFrame, y: pd.Series, params: Params):
    model = KMeans(random_state=0)
    logging.info(f"Fitting model: {model.__class__.__name__}")
    for n_clusters in range(params.kmeans['min_n_clusters'], params.kmeans['max_n_clusters']):
        logging.info(f"Number of clusters: {n_clusters}")
        model.n_clusters = n_clusters
        prediction = model.fit_predict(x)
        x_labelled = x.copy()
        x_labelled['user_class'] = y
        x_labelled['cluster_id'] = prediction
        for cluster_n in range(n_clusters):
            individuals_in_cluster = x_labelled[x_labelled['cluster_id'] == cluster_n]
            support = len(individuals_in_cluster) / len(x) * 100
            # individuals_in_cluster = pd.concat([individuals_in_cluster, y], axis=1, join='inner')
            lows = individuals_in_cluster[individuals_in_cluster['user_class'] == 0]
            highs = individuals_in_cluster[individuals_in_cluster['user_class'] == 1]
            lows_pct = len(lows) / len(individuals_in_cluster) * 100
            highs_pct = len(highs) / len(individuals_in_cluster) * 100
            logging.info(f"Algorithm [{model.__class__.__name__}]. N_of_clusters [{n_clusters}]. Cluster_n. [{cluster_n}]. Supp. [{support:.2f}%]. Lows [{lows_pct:.2f}%]. Highs [{highs_pct:.2f}%].")


def fit_db_scan(x: pd.DataFrame, y: pd.Series, params: Params):
    model = DBSCAN(min_samples=1)
    logging.info(f"Fitting model: {model.__class__.__name__}")
    for eps_value in params.db_scan['eps']:
        logging.info(f"EPS: {eps_value}")
        model.eps = eps_value
        prediction = model.fit_predict(x)
        x_labelled = x.copy()
        x_labelled['user_class'] = y
        x_labelled['cluster_id'] = prediction
        n_clusters = len(set(prediction)) - (1 if -1 in prediction else 0)
        for cluster_n in range(n_clusters):
            individuals_in_cluster = x_labelled[x_labelled['cluster_id'] == cluster_n]
            support = len(individuals_in_cluster) / len(x) * 100
            # individuals_in_cluster = pd.concat([individuals_in_cluster, y], axis=1, join='inner')
            lows = individuals_in_cluster[individuals_in_cluster['user_class'] == 0]
            highs = individuals_in_cluster[individuals_in_cluster['user_class'] == 1]
            lows_pct = len(lows) / len(individuals_in_cluster) * 100
            highs_pct = len(highs) / len(individuals_in_cluster) * 100
            logging.info(f"Algorithm [{model.__class__.__name__}]. EPS [{eps_value}]. N_of_clusters [{n_clusters}]. Cluster_n. [{cluster_n}]. Supp. [{support:.2f}%]. Lows [{lows_pct:.2f}%]. Highs [{highs_pct:.2f}%].")


def fit_optics(x: pd.DataFrame, y: pd.Series, params: Params):
    model = OPTICS(min_samples=1)
    logging.info(f"Fitting model: {model.__class__.__name__}")
    for min_samples in params.optics['min_samples']:
        logging.info(f"Min samples: {min_samples}")
        model.min_samples = min_samples
        prediction = model.fit_predict(x)
        x_labelled = x.copy()
        x_labelled['user_class'] = y
        x_labelled['cluster_id'] = prediction
        n_clusters = len(set(prediction)) - (1 if -1 in prediction else 0)
        for cluster_n in range(n_clusters):
            individuals_in_cluster = x_labelled[x_labelled['cluster_id'] == cluster_n]
            support = len(individuals_in_cluster) / len(x) * 100
            # individuals_in_cluster = pd.concat([individuals_in_cluster, y], axis=1, join='inner')
            lows = individuals_in_cluster[individuals_in_cluster['user_class'] == 0]
            highs = individuals_in_cluster[individuals_in_cluster['user_class'] == 1]
            lows_pct = len(lows) / len(individuals_in_cluster) * 100
            highs_pct = len(highs) / len(individuals_in_cluster) * 100
            logging.info(f"Algorithm [{model.__class__.__name__}]. Min. samples [{min_samples}]. N_of_clusters [{n_clusters}]. Cluster_n. [{cluster_n}]. Supp. [{support:.2f}%]. Lows [{lows_pct:.2f}%]. Highs [{highs_pct:.2f}%].")



if __name__ == '__main__':
    # Load the data for the experiment
    args = parser.parse_args()

    # Set the path for the dataset to explore.
    dataset_csv_path = os.path.join(args.data_dir, 'dataset_and_target.csv')
    assert os.path.isfile(dataset_csv_path), "No dataset found at {}".format(dataset_csv_path)

    # Load the parameters from the experiment params.json file in model_dir
    json_path = os.path.join(args.model_dir, 'params.json')
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'clustering_fit.log'))

    # Load the dataset
    logging.info(f"Loading dataset from {dataset_csv_path}.")
    x, y, features, percentage_features, target = load_dataset_from_csv(args.data_dir)

    # OneHot encoding.
    logging.info(f"Applying OneHot encoder...")
    x, y, features = normalize_datatypes(x, y)
    y.replace({"high": 1, "low": 0}, inplace=True)
    logging.info(f"Shapes after OneHot encoding: features {x.shape}, target {len(y)}.")

    # Scaling to range 0...1.
    logging.info(f"Applying Scaler...")
    x = scale_data_to_range_0_1(x, features, percentage_features)

    # Train the kmeans model
    logging.info("Starting to fit KMeans...")
    fit_kmeans(x, y, params)
    fit_db_scan(x, y, params)
    fit_optics(x, y, params)