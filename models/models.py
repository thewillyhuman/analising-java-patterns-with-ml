import logging

import numpy as np
import pandas as pd
import sklearn.cluster
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from yellowbrick.cluster import KElbowVisualizer

from models.evaluation import evaluate_logistic_regression
from models.utils import Params
import scipy.stats as st


def search_kmeans_elbow(x: pd.DataFrame, from_n_clusters: int, to_n_clusters: int) -> int:
    tmp_model = KMeans(random_state=0)
    visualizer = KElbowVisualizer(tmp_model, k=(from_n_clusters, to_n_clusters), timings=True)
    visualizer.fit(x)
    visualizer.finalize()

    return int(visualizer.elbow_value_)


def build_kmeans_model(params: Params = None) -> KMeans:
    """ Builds a KMeans model.

    :param params:
    :param n_jobs:
    :return:
    """
    if params is None:
        return KMeans(
            n_clusters=2,
            random_state=0
        )
    else:
        return KMeans(
            n_clusters=params.n_clusters,
            random_state=0
        )


def build_elastic_log_reg_model(params: Params = None, n_jobs: int = -1) -> LogisticRegression:
    """Builds a logistic regression with elasticnet penalty with default or given params

    :param params:
    :param n_jobs:
    :return:
    """
    if params is None:
        return LogisticRegression(solver='saga',
                                  penalty='elasticnet',
                                  l1_ratio=0.0,
                                  n_jobs=n_jobs,
                                  random_state=0,
                                  max_iter=1000)
    else:
        return LogisticRegression(solver='saga',
                                  penalty='elasticnet',
                                  C=params.C,
                                  l1_ratio=params.l1_ratio,
                                  n_jobs=n_jobs,
                                  random_state=0,
                                  max_iter=1000)


def fit_and_evaluate_kmeans(kmeans: KMeans, x: pd.DataFrame, y:pd.Series, target_name: str, confidence: float = 0.95) -> (KMeans, pd.DataFrame):
    prediction = kmeans.fit_predict(x)
    x_labelled = x.copy()
    x_labelled['cluster_id'] = prediction

    clusters_data = []
    for cluster_n in range(kmeans.n_clusters):
        individuals_in_cluster = x_labelled[x_labelled['cluster_id'] == cluster_n]
        support = len(individuals_in_cluster) / len(x) * 100
        individuals_in_cluster = pd.concat([individuals_in_cluster, y], axis=1, join='inner')
        lows = individuals_in_cluster[individuals_in_cluster[target_name] == 0]
        highs = individuals_in_cluster[individuals_in_cluster[target_name] == 1]
        lows_pct = len(lows) / len(individuals_in_cluster) * 100
        highs_pct = len(highs) / len(individuals_in_cluster) * 100

        means = individuals_in_cluster.describe().loc['mean'].values.tolist()
        std_devs = individuals_in_cluster.describe().loc['std'].values.tolist()


        #confidence_intervals = [np.percentile(individuals_in_cluster[feature_name], [100 * (1 - confidence) / 2, 100 * (1 - (1 - confidence) / 2)]).tolist() for feature_name in individuals_in_cluster.columns.values.tolist()]
        confidence_intervals = [st.norm.interval(alpha=confidence, loc=np.mean(individuals_in_cluster[feature_name]), scale=st.sem(individuals_in_cluster[feature_name])) for feature_name in individuals_in_cluster.columns.values.tolist()]
        confidence_intervals = [(f"{ci[0]:.3f}", f"{mean:.3f}", f"{ci[1]:.3f}") for ci, mean in zip(confidence_intervals, means)]
        #confidence_intervals = [(f"{(mean-(0.05)):.2f}", f"{mean:.2f}", f"{(mean + (0.05)):.2f}") for mean, std_dev in zip(means, std_devs)]

        cluster_data = [support, lows_pct, highs_pct] + confidence_intervals
        clusters_data.append(cluster_data)

    results_df_columns = ["Support", "Low", "High"] + individuals_in_cluster.columns.values.tolist()
    results_df = pd.DataFrame(clusters_data, columns=results_df_columns)
    return kmeans, results_df


def train_and_evaluate_log_reg(log_reg: LogisticRegression, x: pd.DataFrame, y: pd.Series) -> (LogisticRegression, float, float, float, float):
    """Fits and evaluates the given logistic regression model

    :param log_reg: is the model to fit. Needs to be parametrized.
    :param x: is the features dataset.
    :param y: is the targets dataset.
    :return: the fitted model, the accuracy, the recall, the precision and the f1 score.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=0)
    log_reg.fit(x_train, y_train)
    acc, rec, pre, f1 = evaluate_logistic_regression(log_reg=log_reg, x_test=x_test, y_test=y_test)

    return log_reg, acc, rec, pre, f1


def save_log_reg_coefs_to_excel(log_reg: LogisticRegression, features: [str], out_file_path: str) -> None:
    tmp_df = pd.DataFrame(log_reg.coef_).T
    tmp_df['features'] = features
    tmp_df.to_excel(excel_writer=out_file_path)