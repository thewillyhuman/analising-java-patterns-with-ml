"""Utility functions to work with scikit-learn KMeans model"""
import logging

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import t
from sklearn.cluster import KMeans
from statsmodels.stats import weightstats as stests
from yellowbrick.cluster import KElbowVisualizer

from models.utils import Params


def build_model(params: Params = None) -> KMeans:
    """ Builds a basic KMeans model with the number of params  n_clusters."""
    return KMeans(n_clusters=params.n_clusters)


def find_elbow(x: pd.DataFrame, from_n_clusters: int, to_n_clusters: int) -> int:
    """Finds and returns the elbow for KMeans model"""
    tmp_model = KMeans()
    visualizer = KElbowVisualizer(tmp_model, k=(from_n_clusters, to_n_clusters), timings=True)
    visualizer.fit(x)
    visualizer.finalize()

    return int(visualizer.elbow_value_)


def fit_and_evaluate(model: KMeans, x: pd.DataFrame, y: pd.Series, target_name: str,
                     confidence: float = 0.95) -> (KMeans, pd.DataFrame):
    """Fits and evaluates a KMeans model."""
    prediction = model.fit_predict(x)
    x_labelled = x.copy()
    x_labelled['cluster_id'] = prediction

    clusters_data = []
    for cluster_n in range(model.n_clusters):
        individuals_in_cluster = x_labelled[x_labelled['cluster_id'] == cluster_n]
        support = len(individuals_in_cluster) / len(x) * 100
        individuals_in_cluster = pd.concat([individuals_in_cluster, y], axis=1, join='inner')
        lows = individuals_in_cluster[individuals_in_cluster[target_name] == 0]
        highs = individuals_in_cluster[individuals_in_cluster[target_name] == 1]
        lows_pct = len(lows) / len(individuals_in_cluster) * 100
        highs_pct = len(highs) / len(individuals_in_cluster) * 100

        means = individuals_in_cluster.describe().loc['mean'].values.tolist()

        confidence_intervals = [
            st.norm.interval(alpha=confidence, loc=np.mean(individuals_in_cluster[feature_name]),
                             scale=st.sem(individuals_in_cluster[feature_name])) for feature_name in
            individuals_in_cluster.columns.values.tolist()]

        confidence_intervals = [f"[{ci[0]:.3f} - {ci[1]:.3f}]" for ci, mean in zip(
            confidence_intervals, means)]

        cluster_data = [support, lows_pct, highs_pct] + confidence_intervals
        clusters_data.append(cluster_data)

    results_df_columns = ["Support", "Low", "High"] + individuals_in_cluster.columns.values.tolist()
    results_df = pd.DataFrame(clusters_data, columns=results_df_columns)
    return model, results_df


def explore_distributions_and_centroids(model: KMeans, X: pd.DataFrame, y: pd.Series,
                                        numeric_feature_names: [str], target_name: str) -> None:
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
