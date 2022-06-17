import argparse
import logging
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, NMF
from sklearn.manifold import TSNE

from build_dataset import download_data, normalize_datatypes, scale_data_to_range_0_1
from models.utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True, help="Directory containing query and features")
parser.add_argument('--model-dir', type=str, required=True, help="Directory containing params.json")
parser.add_argument('--database-name', type=str, default='patternminingV2')


if __name__ == '__main__':

    # Load the data for the experiment
    args = parser.parse_args()
    query_path = os.path.join(args.data_dir, 'query.sql')
    features_path = os.path.join(args.data_dir, 'features.txt')
    target_path = os.path.join(args.data_dir, 'target.txt')
    percentage_features_path = os.path.join(args.data_dir, 'percentage_features.txt')

    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)

    figure_path = os.path.join(args.model_dir, 'visualization.svg')
    assert os.path.isfile(query_path), "No json configuration file found at {}".format(query_path)
    assert os.path.isfile(features_path), "No features file found at {}".format(features_path)
    assert os.path.isfile(percentage_features_path), "No percentage features file found at {}".format(percentage_features_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

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
    # x = scale_data_to_range_0_1(x, features, percentage_features)

    if len(x) > 50000:
        logging.info(f"Sampling data to 50.000 instances before ploting")
        x = x.sample(n=50000, random_state=0)
        y = y[x.index]

        # Init the figure that will hold all subplots.
    figure, axis = plt.subplots(2, 2)
    figure.set_figwidth(9)
    figure.set_figheight(9)

    # PCA
    tmp_model = PCA(n_components=2)
    X_embedded = tmp_model.fit_transform(x)
    explained_variance = np.cumsum(tmp_model.explained_variance_ratio_)[1]  # dimention 2 (it is cumulative sum)
    # plot it
    axis[0, 0].set_title(f"Linear PCA. Exp.var:{explained_variance:.4f}%.")
    axis[0, 0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)

    # TSNE
    tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
    X_embedded = tmp_model.fit_transform(x)
    # plot it
    axis[0, 1].set_title(f"t-SNE (learning_rate=200).")
    axis[0, 1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)

    # KernelPCA
    tmp_model = KernelPCA(n_components=2, kernel="rbf")
    X_embedded = tmp_model.fit_transform(x)
    # plot it
    axis[1, 0].set_title(f"KernelPCA (kernel=rbf).")
    axis[1, 0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)

    # NMF
    tmp_model = NMF(n_components=2, init='random', random_state=0, max_iter=5000)
    X_embedded = tmp_model.fit_transform(x)
    # plot it
    axis[1, 1].set_title(f"NMF")
    axis[1, 1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)

    figure.savefig(figure_path)
