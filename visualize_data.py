import argparse
import logging
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, NMF
from sklearn.manifold import TSNE

from build_dataset import normalize_datatypes, scale_data_to_range_0_1
from models.utils import set_logger, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True,
                    help="Directory containing query and features")
parser.add_argument('--out-dir', type=str, required=True)

if __name__ == '__main__':

    # Load the data for the experiment
    args = parser.parse_args()
    dataset = Dataset(data_dir=args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    figure_path = os.path.join(args.out_dir, 'visualization.svg')

    # Set the logger
    set_logger(os.path.join(args.out_dir, 'visualize.log'))

    # Load the dataset
    x, y = dataset.x, dataset.y
    logging.info(f"Dataset loaded. Shape: {x.shape}, {y.shape}.")

    # OneHot encoding
    x, y, features = normalize_datatypes(x, y)
    logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

    # Scale features to range 0 1
    x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
    logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

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
    explained_variance = np.cumsum(tmp_model.explained_variance_ratio_)[1]
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
    logging.info("Data saved.")
