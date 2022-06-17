import argparse
import logging
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA, NMF
from sklearn.manifold import TSNE

from build_dataset import download_data, normalize_datatypes
from models.utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True, help="Directory containing query and features")
parser.add_argument('--out-dir', type=str, required=True, help="Directory containing params.json")
parser.add_argument('--database-name', type=str, default='patternminingV2')
parser.add_argument('--from-k', type=int, default=2)
parser.add_argument('--to-k', type=int, default=10)


if __name__ == '__main__':

    # Load the data for the experiment
    args = parser.parse_args()
    query_path = os.path.join(args.data_dir, 'query.sql')
    features_path = os.path.join(args.data_dir, 'features.txt')
    target_path = os.path.join(args.data_dir, 'target.txt')
    percentage_features_path = os.path.join(args.data_dir, 'percentage_features.txt')

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    figure_path = os.path.join(args.out_dir, 'visualization.svg')
    assert os.path.isfile(query_path), "No json configuration file found at {}".format(query_path)
    assert os.path.isfile(features_path), "No features file found at {}".format(features_path)
    assert os.path.isfile(percentage_features_path), "No percentage features file found at {}".format(percentage_features_path)

    # Set the logger
    set_logger(os.path.join(args.out_dir, 'elbow_generation.log'))

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

    logging.info(f"Exploring KMeans space from k={args.from_k} to k={args.to_k}.")
    inertias = []
    for n_clusters in range(args.from_k, args.to_k):
        logging.info(f"Exploring KMeans with k={n_clusters}.")
        model = KMeans(n_clusters=n_clusters, random_state=0)
        model.fit(x)
        inertias.append(model.inertia_)

    cluster_numbers = np.arange(args.from_k, args.to_k, 1)
    fig, ax = plt.subplots()
    ax.plot(cluster_numbers, inertias, marker='x')
    plt.xticks(np.arange(min(cluster_numbers), max(cluster_numbers) + 1, 1.0))

    ax.set(xlabel='Number of clusters', ylabel='Inertia')

    ax.annotate('Elbow at k=4', xy=(4, inertias[2]), xytext=(5, 1.4e9),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax.grid()

    fig.savefig("elbow_kmeans_het3.svg")
    #plt.show()