import argparse
import logging
import os

import numpy as np
import scipy.stats as st

from joblib import load
from matplotlib import pyplot as plt

from build_dataset import normalize_datatypes, scale_data_to_range_0_1
from models.utils import Dataset, set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--model-dir', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--features', nargs='+', required=True)
parser.add_argument('--features-titles', nargs='+', required=True)
parser.add_argument('--vertical-charts', type=int, required=True)
parser.add_argument('--horizontal-charts', type=int, required=True)

if __name__ == '__main__':

    # Load the data for the experiment
    args = parser.parse_args()
    dataset = Dataset(data_dir=args.data_dir)
    model_path = os.path.join(args.model_dir, 'model.joblib')
    out_path = os.path.join(args.out_dir, 'centroids_visualization.svg')

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'visualize_cluster_centroids.log'))

    # Load the model
    model = load(model_path)
    logging.info(f"Model loaded.")
    print(model)

    # Load the dataset
    x, y, features = dataset.x, dataset.y, dataset.features
    logging.info(f"Dataset loaded. Shape: {x.shape}, {y.shape}.")

    # OneHot encoding
    x, y, features = normalize_datatypes(x, y)
    logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

    # Scale features to range 0 1
    # x_scalled = scale_data_to_range_0_1(x.copy(), features, dataset.percentage_features)
    # logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

    processable_features = list((feature for feature in args.features if feature in features))

    assert len(processable_features) == args.vertical_charts * args.horizontal_charts

    prediction = model.predict(x)
    x_labelled = x.copy()
    x_labelled['cluster_id'] = prediction

    figure, axis = plt.subplots(args.vertical_charts, args.horizontal_charts)
    figure.set_figwidth(5*args.horizontal_charts)
    figure.set_figheight(5*args.vertical_charts)

    figure_row = 0
    figure_col = 0
    for feature, title in zip(processable_features, args.features_titles):
        logging.info(f"Processing feature {feature}.")
        means = []
        mins = []
        maxs = []
        for n_custer in range(model.n_clusters):
            individuals_in_cluster = x_labelled[x_labelled['cluster_id'] == n_custer]
            individuals_for_feature = individuals_in_cluster[feature]
            mean = np.mean(individuals_for_feature)
            means.append(mean)
            cis = st.norm.interval(alpha=0.95, loc=np.mean(individuals_for_feature),scale=st.sem(individuals_for_feature))
            print(cis[0], mean, cis[1])
            mins.append(mean - cis[0])
            maxs.append(cis[1] - mean)
        #plt.scatter(means, range(1, model.n_clusters + 1), c="blue")
        x_error = [mins, maxs]
        axis[figure_row, figure_col].errorbar(means, range(1, model.n_clusters + 1), xerr=x_error, fmt='o', color='k', capsize=2)
        axis[figure_row, figure_col].set_ylabel("Cluster Number")
        y_labels = [f"C{n}" for n in range(1, model.n_clusters + 1)]
        axis[figure_row, figure_col].set_yticks(range(1, model.n_clusters + 1), labels=y_labels)
        axis[figure_row, figure_col].set_title(f"{title}")
        axis[figure_row, figure_col].axvline(x=np.mean(means), color='orange', linestyle='--')

        figure_col = figure_col + 1
        if figure_col == args.horizontal_charts:
            figure_row = figure_row + 1
            figure_col = 0

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    figure.savefig(out_path, dpi=1200, bbox_inches='tight')