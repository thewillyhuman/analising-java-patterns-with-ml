import argparse
import logging
import os

import pandas as pd
import sqlalchemy

from models.utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True, help="Base data directory.")

DATABASE_IP = '156.35.94.139'
DATABASE_PORT = '5432'
DATABASE_USERNAME = 'postgres'
DATABASE_PASSWORD = 'postgres'


def load_dataset_from_csv(data_dir: str) -> (pd.DataFrame, pd.Series, [str], str):
    dataset_path = os.path.join(data_dir, 'dataset_and_target.csv')
    features_path = os.path.join(data_dir, 'features.txt')
    percentage_features_path = os.path.join(data_dir, 'percentage_features.txt')
    target_path = os.path.join(data_dir, 'target.txt')

    assert os.path.isfile(dataset_path), f"No dataset fount at {dataset_path}"
    assert os.path.isfile(features_path), f"No features found at {features_path}"
    assert os.path.isfile(
        percentage_features_path), f"No percentage features found at {percentage_features_path}"
    assert os.path.isfile(target_path), f"No target found at {target_path}"

    features = [feature.rstrip() for feature in open(features_path, mode='r').readlines()]
    percentage_features = [feature.rstrip() for feature in
                           open(percentage_features_path, mode='r').readlines()]
    target = open(target_path, mode='r').read()

    full_table = pd.read_csv(dataset_path)
    return full_table[features], full_table[target], features, percentage_features, target


def download_data(query: str, database_name: str, features: [str], target: str) -> (
        pd.DataFrame, pd.Series):
    # Connect to the database.
    db_conn_str = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_IP}:{DATABASE_PORT}/{database_name}"
    db_conn = sqlalchemy.create_engine(db_conn_str)

    # Split features and target
    full_table = pd.read_sql_query(sql=query, con=db_conn)
    return full_table[features], full_table[[target]].iloc[:, 0]


def normalize_datatypes(x: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.Series, [str]):
    x = pd.get_dummies(x)
    x = x.astype('float32')
    y = y.apply(lambda value: 0 if value == "low" else 1)  # high will be 1 and low will be 0.
    y = y.astype('float32')
    x = x.fillna(0.0)
    columns_names = x.columns.tolist()
    return x, y, columns_names


def scale_data_to_range_0_1(x: pd.DataFrame, feature_names: [str],
                            percentage_feature_names: [str]) -> pd.DataFrame:
    for column in feature_names:
        if column in percentage_feature_names:
            x[column] = x[column] / 100.0
        else:
            x[column] = x[column] / x[column].max()
    x = x.fillna(0.0)
    return x


def build_dataset(database_name: str, data_dir: str) -> None:
    # Load paths from data directory.
    query_path = os.path.join(data_dir, 'query.sql')
    features_path = os.path.join(data_dir, 'features.txt')
    target_path = os.path.join(data_dir, 'target.txt')

    # Ensure paths exists.
    assert os.path.isfile(query_path), f"No json configuration file found at {query_path}"
    assert os.path.isfile(features_path), f"No json configuration file found at {features_path}"
    assert os.path.isfile(target_path), f"No json configuration file found at {target_path}"

    # Download the dataset from the given database.
    logging.info(f"Downloading dataset [{data_dir}]...")
    query = open(query_path, mode='r').read()
    features = [feature.rstrip() for feature in open(features_path, mode='r').readlines()]
    target = open(target_path, mode='r').read()
    x, y = download_data(query=query, database_name=database_name, features=features, target=target)
    logging.info(f"Dataset downloaded. Features shape {x.shape}. Target shape {y.shape}.")

    logging.info("Saving dataset to a single csv.")
    x[target] = y
    out_file_path = os.path.join(data_dir, 'dataset_and_target.csv')
    x.to_csv(out_file_path, index=False)
    logging.info("Dataset saved.")


if __name__ == '__main__':
    # Load the data for the experiment
    args = parser.parse_args()

    # Check for data dir.
    assert os.path.isdir(args.data_dir), f"No data dir for data at {args.data_dir}."

    # Create and check that base and unique data dirs exist.
    base_data_dir = os.path.join(args.data_dir, 'base')
    unique_data_dir = os.path.join(args.data_dir, 'unique')
    assert os.path.isdir(base_data_dir), f"No base dir for data at {base_data_dir}."
    assert os.path.isdir(unique_data_dir), f"No unique dir for data at {unique_data_dir}."

    # Set the logger at data_dir
    set_logger(os.path.join(args.data_dir, 'build_dataset.log'))

    for subdir in os.listdir(base_data_dir):
        subdir = os.path.join(base_data_dir, subdir)
        expected_data_dir = os.path.join(subdir, 'data_and_target.csv')
        build_dataset('patternmining', subdir)

    for subdir in os.listdir(unique_data_dir):
        subdir = os.path.join(unique_data_dir, subdir)
        expected_data_dir = os.path.join(subdir, 'data_and_target.csv')
        build_dataset('patternminingV2', subdir)
