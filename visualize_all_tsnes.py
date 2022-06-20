import argparse
import logging
import os

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from build_dataset import normalize_datatypes, scale_data_to_range_0_1
from models.utils import set_logger, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--orientation', type=str, required=True)

DATASETS_PATHS = [
    'data/base/type_defs',
    'data/base/field_defs',
    'data/base/method_defs',
    'data/base/types',
    'data/base/statements',
    'data/base/expressions',
    'data/base/het1',
    'data/base/het2',
    'data/base/het3',
    'data/base/het4',
    'data/base/het5',
]

PLOTS_TITLES = [
    'Type Definitions (86.66%)',
    'Field Definitions (47.81%)',
    'Method Definitions (91.83%)',
    'Types (75.56%)',
    'Statements (67.68%)',
    'Expressions (82.29%)',
    'Heterogeneous 1 (84.73%)',
    'Heterogeneous 2 (79.15%)',
    'Heterogeneous 3 (82.05%)',
    'Heterogeneous 4 (64.85%)',
    'Heterogeneous 5 (61.23%)',
]

N_COLS = 3
N_ROWS = 4

if __name__ == '__main__':

    # Load the data for the experiment
    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'tsnes_visualization.png')

    # Set the logger
    set_logger(os.path.join(args.out_dir, 'tsnes_visualization.log'))

    if args.orientation == 'vertical':
        figure, axis = plt.subplots(N_ROWS, N_COLS)
        figure.set_figwidth(12)
        figure.set_figheight(16)

        figure_row = 0
        figure_col = 0
        iteration = 0
        for data_dir in DATASETS_PATHS:
            logging.info(f"Computing t-SNE for {data_dir}")
            if figure_col == 2 and figure_row == 0:
                figure_col = 0
                figure_row = figure_row + 1
            # dataset = Dataset(data_dir=data_dir)
            dataset = Dataset(data_dir=data_dir)

            # Load the dataset
            x, y = dataset.x, dataset.y
            logging.info(f"Dataset loaded. Shape: {x.shape}, {y.shape}.")

            if len(x) > 50000:
                logging.info(f"Sampling data to 50.000 instances before ploting")
                x = x.sample(n=50000, random_state=0)
                y = y[x.index]

            # OneHot encoding
            x, y, features = normalize_datatypes(x, y)
            logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

            # Scale features to range 0 1
            x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
            logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

            # TSNE
            tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
            X_embedded = tmp_model.fit_transform(x)
            # plot it
            axis[figure_row, figure_col].set_title(PLOTS_TITLES[iteration])
            axis[figure_row, figure_col].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)

            iteration = iteration + 1
            figure_col = figure_col + 1
            if figure_col == 3:
                figure_row = figure_row + 1
                figure_col = 0
    else:
        N_COLS = 4
        N_ROWS = 3
        figure, axis = plt.subplots(N_ROWS, N_COLS)
        figure.set_figwidth(16)
        figure.set_figheight(12)

        figure_row = 0
        figure_col = 0
        iteration = 0
        for data_dir in DATASETS_PATHS:
            logging.info(f"Computing t-SNE for {data_dir}")
            if figure_col == 3 and figure_row == 0:
                figure_col = 0
                figure_row = figure_row + 1
            # dataset = Dataset(data_dir=data_dir)
            dataset = Dataset(data_dir=data_dir)

            # Load the dataset
            x, y = dataset.x, dataset.y
            logging.info(f"Dataset loaded. Shape: {x.shape}, {y.shape}.")

            if len(x) > 50000:
                logging.info(f"Sampling data to 50.000 instances before ploting")
                x = x.sample(n=50000, random_state=0)
                y = y[x.index]

            # OneHot encoding
            x, y, features = normalize_datatypes(x, y)
            logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

            # Scale features to range 0 1
            x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
            logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

            # TSNE
            tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
            X_embedded = tmp_model.fit_transform(x)
            # plot it
            axis[figure_row, figure_col].set_title(PLOTS_TITLES[iteration])
            axis[figure_row, figure_col].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)

            iteration = iteration + 1
            figure_col = figure_col + 1
            if figure_col == 4:
                figure_row = figure_row + 1
                figure_col = 0



    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    figure.savefig(out_path, dpi=1200, bbox_inches='tight')
    #plt.show()