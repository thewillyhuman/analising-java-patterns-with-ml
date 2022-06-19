"""General utility functions and classes"""

import json
import logging
import os

import pandas as pd


class Dataset:
    """Class that loads a dataset from a directory.
    Example:
    ```
    dataset = Dataset(data_dir)
    print(dataset.x)
    print(dataset.y)
    print(dataset.features)
    """

    def __init__(self, data_dir: str) -> None:
        self.dataset_path_ = os.path.join(data_dir, 'dataset_and_target.csv')
        self.features_path_ = os.path.join(data_dir, 'features.txt')
        self.percentage_features_path_ = os.path.join(data_dir, 'percentage_features.txt')
        self.numeric_features_path_ = os.path.join(data_dir, 'numeric_features.txt')
        self.target_path_ = os.path.join(data_dir, 'target.txt')

        # Check every file exists.
        assert os.path.isfile(self.features_path_)
        assert os.path.isfile(self.target_path_)
        assert os.path.isfile(self.numeric_features_path_)
        assert os.path.isfile(self.percentage_features_path_)
        assert os.path.isfile(self.dataset_path_)

        self.features = None  # String list features.
        self.target = None  # String target column name.
        self.numeric_features = None  # String list numeric features.
        self.percentage_features = None  # String list percentage features.
        self.dataset = None  # Full dataset.
        self.x = None  # Dataset without target.
        self.y = None  # Dataset target.

        self.update()

    def update(self) -> None:
        """Updates all Dataset parameters from the Dataset instance"""

        # Load features.
        with open(self.features_path_) as f:
            self.features = [feature.rstrip() for feature in f.readlines()]

        # Load target.
        with open(self.target_path_) as f:
            self.target = f.read()

        # Load numeric features.
        with open(self.numeric_features_path_) as f:
            self.numeric_features = [feature.rstrip() for feature in f.readlines()]

        # Load percentage features.
        with open(self.percentage_features_path_) as f:
            self.percentage_features = [feature.rstrip() for feature in f.readlines()]

        self.dataset = pd.read_csv(self.dataset_path_)
        self.x = self.dataset[self.features].copy()
        self.y = self.dataset[self.target].copy()


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
