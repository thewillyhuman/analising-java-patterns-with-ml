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
    'data/base/programs',
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
    'Programs (98.64%)',
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

# Load the data for the experiment
args = parser.parse_args()

# Set the logger
set_logger(os.path.join(args.out_dir, 'tsnes_visualization.log'))

# Create figure 1
figure, axis = plt.subplots(2, 2)
figure.set_figwidth(12)
figure.set_figheight(12)

# Load the dataset 0
dataset = Dataset(data_dir=DATASETS_PATHS[0])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[0, 0].set_title(PLOTS_TITLES[0])
axis[0, 0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)

# Load the dataset 1
dataset = Dataset(data_dir=DATASETS_PATHS[1])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[0, 1].set_title(PLOTS_TITLES[1])
axis[0, 1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)


# Load the dataset 2
dataset = Dataset(data_dir=DATASETS_PATHS[2])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[1, 0].set_title(PLOTS_TITLES[2])
axis[1, 0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)


# Load the dataset 1
dataset = Dataset(data_dir=DATASETS_PATHS[3])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[1, 1].set_title(PLOTS_TITLES[3])
axis[1, 1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)

out_path = os.path.join(args.out_dir, 'tsnes_visualization_1.png')
plt.subplots_adjust(wspace=0.3, hspace=0.3)
figure.savefig(out_path, dpi=600, bbox_inches='tight')




# Create figure 2
figure, axis = plt.subplots(3, 2)
figure.set_figwidth(12)
figure.set_figheight(18)


# Load the dataset 1
dataset = Dataset(data_dir=DATASETS_PATHS[4])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[0, 0].set_title(PLOTS_TITLES[4])
axis[0, 0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)


# Load the dataset 1
dataset = Dataset(data_dir=DATASETS_PATHS[5])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[0, 1].set_title(PLOTS_TITLES[5])
axis[0, 1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)


# Load the dataset 1
dataset = Dataset(data_dir=DATASETS_PATHS[6])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[1, 0].set_title(PLOTS_TITLES[6])
axis[1, 0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)


# Load the dataset 1
dataset = Dataset(data_dir=DATASETS_PATHS[7])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[1, 1].set_title(PLOTS_TITLES[7])
axis[1, 1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)



# Load the dataset 1
dataset = Dataset(data_dir=DATASETS_PATHS[8])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[2, 0].set_title(PLOTS_TITLES[8])
axis[2, 0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)


# Load the dataset 1
dataset = Dataset(data_dir=DATASETS_PATHS[9])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[2, 1].set_title(PLOTS_TITLES[9])
axis[2, 1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)


out_path = os.path.join(args.out_dir, 'tsnes_visualization_2.png')
plt.subplots_adjust(wspace=0.3, hspace=0.3)
figure.savefig(out_path, dpi=600, bbox_inches='tight')







# Create figure 2
figure, axis = plt.subplots(1, 2)
figure.set_figwidth(12)
figure.set_figheight(6)



# Load the dataset 1
dataset = Dataset(data_dir=DATASETS_PATHS[10])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[0].set_title(PLOTS_TITLES[10])
axis[0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)


# Load the dataset 1
dataset = Dataset(data_dir=DATASETS_PATHS[11])
x, y = dataset.x, dataset.y

if len(x) > 10:
    x = x.sample(n=10, random_state=0)
    y = y[x.index]

# OneHot encoding
x, y, features = normalize_datatypes(x, y)
logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

# Scale features to range 0 1
x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

tmp_model = TSNE(n_components=2, init='random', learning_rate=200, random_state=0)
X_embedded = tmp_model.fit_transform(x)
# plot it
axis[1].set_title(PLOTS_TITLES[11])
axis[1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', alpha=0.2)


out_path = os.path.join(args.out_dir, 'tsnes_visualization_3.png')
plt.subplots_adjust(wspace=0.3, hspace=0.3)
figure.savefig(out_path, dpi=600, bbox_inches='tight')