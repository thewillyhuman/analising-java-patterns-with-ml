import argparse
import logging
import os

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

from build_dataset import normalize_datatypes
from models.utils import set_logger, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--from-k', type=int, default=2)
parser.add_argument('--to-k', type=int, default=15)
parser.add_argument('--plot-timings', action='store_true')
parser.add_argument('--metric', type=str, default='distortion')

if __name__ == '__main__':
    # Load the data for the experiment
    args = parser.parse_args()
    dataset = Dataset(data_dir=args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    figure_path = os.path.join(args.out_dir, 'visualization.svg')

    # Set the logger
    set_logger(os.path.join(args.out_dir, 'elbow_generation.log'))

    # Load the dataset
    x, y = dataset.x, dataset.y
    logging.info(f"Dataset loaded. Shape: {x.shape}, {y.shape}.")

    # OneHot encoding
    x, y, features = normalize_datatypes(x, y)
    logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

    logging.info(f"Exploring KMeans space from k={args.from_k} to k={args.to_k}.")

    model = KMeans(random_state=0)
    visualizer = KElbowVisualizer(
        model, k=(args.from_k, args.to_k), metric=args.metric, timings=args.plot_timings
    )
    visualizer.fit(x)
    visualizer.finalize()
    plt.savefig(figure_path)
