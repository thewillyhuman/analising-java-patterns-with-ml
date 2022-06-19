import argparse
import logging
import os

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import t
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from statsmodels.stats import weightstats as stests
from tabulate import tabulate

from build_dataset import download_data, normalize_datatypes
from models.models import build_kmeans_model, fit_and_evaluate_kmeans, search_kmeans_elbow
from models.utils import Params, set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True,
                    help="Directory containing query and features")
parser.add_argument('--model-dir', type=str, required=True, help="Directory containing params.json")
parser.add_argument('--compute-elbow', action='store_true',
                    help="Whether to compute the number of clusters or not")
parser.add_argument('--max-n-features', type=int, default=-1,
                    help="Máximum number of features to select. If present will use feature selection.")
parser.add_argument('--database-name', type=str, default='patternminingV2')


def explore_kmeans_distributions_and_centroids(model: KMeans, X: pd.DataFrame,
                                               y: pd.Series, numeric_feature_names: [str],
                                               target_name: str) -> None:
    # KMeans model created and fitted with the number of clsuters resulted from the elbow.
    kmeans_model = model
    X_labels = kmeans_model.fit_predict(X)
    X_labelled = X.copy()
    X_labelled['cluster_id'] = X_labels

    # Temporal array to store the supports (% population) that clusters associated with an expertise level have.
    supports_of_clusters_associated_with_an_expertise_level = []

    # Explore each one of the clusters (supports and tukey test).
    for current_cluster_index in range(kmeans_model.n_clusters):
        # Cluster support computing.
        current_cluster_individuals = X_labelled[X_labelled['cluster_id'] == current_cluster_index]
        support = len(current_cluster_individuals) / len(X) * 100
        current_cluster_individuals = pd.concat([current_cluster_individuals, y], axis=1,
                                                join='inner')
        lows = current_cluster_individuals[current_cluster_individuals[target_name] == 0]
        highs = current_cluster_individuals[current_cluster_individuals[target_name] == 1]
        lows_pct = len(lows) / len(current_cluster_individuals) * 100
        highs_pct = len(highs) / len(current_cluster_individuals) * 100

        if lows_pct > 70.0 or highs_pct > 70.0:
            logging.info(
                f"Cluster [{current_cluster_index}]. Support [{support:.4f}]. Lows [{lows_pct:.4f}]. Highs "
                f"[{highs_pct:.4f}]. (*)")
            supports_of_clusters_associated_with_an_expertise_level.append(support)
        else:
            logging.info(
                f"Cluster [{current_cluster_index}]. Support [{support:.4f}]. Lows [{lows_pct:.4f}]. Highs "
                f"[{highs_pct:.4f}].")

        # Tukey test to identify variables that identify the cluster.
        for current_feature in numeric_feature_names:

            p_values_less_than_0_05 = []
            for other_cluster_index in range(kmeans_model.n_clusters):
                other_cluster_individuals = X_labelled[
                    X_labelled['cluster_id'] == other_cluster_index]

                if current_cluster_index != other_cluster_index:
                    column_from_current_cluster = current_cluster_individuals[current_feature]
                    column_from_other_cluster = other_cluster_individuals[current_feature]
                    ttest, p_value, df = stests.ttest_ind(column_from_current_cluster,
                                                          column_from_other_cluster)
                    p_values_less_than_0_05.append(p_value < 0.05)

            m = current_cluster_individuals[current_feature].mean()
            s = current_cluster_individuals[current_feature].std()
            dof = len(current_cluster_individuals[current_feature]) - 1
            confidence = 0.95
            t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
            low_ci, high_ci = (
                m - s * t_crit / np.sqrt(len(current_cluster_individuals[current_feature])),
                m + s * t_crit / np.sqrt(len(current_cluster_individuals[current_feature])))
            if sum(p_values_less_than_0_05) == len(p_values_less_than_0_05):
                logging.info(
                    f"\tIdentified by feature [{current_feature}]. Range [{low_ci:.3f} - {high_ci:.3f}].")

    logging.info(f"Support of clusters associated with a level of experience:"
                 f" {sum(supports_of_clusters_associated_with_an_expertise_level):.4f}.")


if __name__ == '__main__':

    # Load the data for the experiment
    args = parser.parse_args()
    query_path = os.path.join(args.data_dir, 'query.sql')
    features_path = os.path.join(args.data_dir, 'features.txt')
    numeric_features_path = os.path.join(args.data_dir, 'numeric_features.txt')
    target_path = os.path.join(args.data_dir, 'target.txt')
    percentage_features_path = os.path.join(args.data_dir, 'percentage_features.txt')
    assert os.path.isfile(query_path), "No json configuration file found at {}".format(query_path)
    assert os.path.isfile(features_path), "No features file found at {}".format(features_path)
    assert os.path.isfile(numeric_features_path), "No numeric features file found at {}".format(
        features_path)
    assert os.path.isfile(
        percentage_features_path), "No percentage features file found at {}".format(
        percentage_features_path)

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
    numeric_features = [feature.rstrip() for feature in
                        open(numeric_features_path, mode='r').readlines()]
    target = open(target_path, mode='r').read()
    percentage_features = [feature.rstrip() for feature in
                           open(percentage_features_path, mode='r').readlines()]
    x, y = download_data(query=query, database_name=args.database_name, features=features,
                         target=target)
    logging.info("Downloaded dataset. Features shape {}. Target shape {}.".format(x.shape, y.shape))

    # OneHot encoding
    x, y, features = normalize_datatypes(x, y)
    logging.info(
        "Features normalized. Features shape {}. Target shape {}.".format(x.shape, y.shape))

    # Scale features to range 0 1
    # x = scale_data_to_range_0_1(x, features, percentage_features)
    logging.info("Features scaled. Features shape {}. Target shape {}.".format(x.shape, y.shape))

    if args.max_n_features != -1:
        selector = SelectKBest(mutual_info_regression, k=args.max_n_features)
        selector.fit(x, y)
        print(x.columns[selector.get_support()].values.tolist())
        x = x[x.columns[selector.get_support()].values.tolist()]
        features = x.columns.values.tolist()

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

    row_index = 0
    for row in model.cluster_centers_:
        cluster_rule = "["
        column_index = 0
        for column in row:
            cluster_rule = cluster_rule + f"{features[column_index]}={column:.8f}"
            if column_index != len(row) - 1:
                cluster_rule = cluster_rule + " ∧ "
            column_index = column_index + 1
        cluster_rule = cluster_rule + "]"

        logging.info("")
        logging.info("")
        logging.info(f"Centroids: {cluster_rule}.")
        logging.info(f"Support: {metrics_df['Support'].iloc[row_index]:.2f}%.")
        logging.info(f"Distribution: {metrics_df['Low'].iloc[row_index]:.2f}% (low),"
                     f" {metrics_df['High'].iloc[row_index]:.2f}% (high).")
        row_index = row_index + 1

    logging.info("")
    logging.info("")
    logging.info("")
    explore_kmeans_distributions_and_centroids(model, x, y, numeric_feature_names=numeric_features,
                                               target_name=target)
    logging.info("")
    logging.info("")
    logging.info("")
    logging.info(
        "Centroids: \n" + tabulate(metrics_df.T, headers='keys', tablefmt='tsv', floatfmt=".2f"))
    logging.info("All saved and finished.")
