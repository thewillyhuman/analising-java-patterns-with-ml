import pandas as pd
import sqlalchemy
import argparse
import logging
import os

from models.utils import set_logger, select_features

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True, help="Directory containing query and features")
parser.add_argument('--database-name', type=str, default='patternminingV2')
parser.add_argument('--out-file-name', type=str, required=True)

DATABASE_IP = '156.35.94.139'
DATABASE_PORT = '5432'
DATABASE_USERNAME = 'postgres'
DATABASE_PASSWORD = 'postgres'


def download_data(query: str, database_name: str, features: [str], target: str) -> (pd.DataFrame, pd.Series):

    # Connect to the database.
    db_conn_str = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_IP}:{DATABASE_PORT}/{database_name}"
    db_conn = sqlalchemy.create_engine(db_conn_str)

    # Split features and target
    full_table = pd.read_sql_query(sql=query, con=db_conn)
    return full_table[features], full_table[[target]].iloc[:, 0]


def normalize_datatypes(x: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.Series, [str]):
    x = pd.get_dummies(x)
    x = x.astype('float32')
    y = y.apply(lambda value: 0 if value == "low" else 1) # high will be 1 and low will be 0.
    y = y.astype('float32')
    x = x.fillna(0.0)
    columns_names = x.columns.tolist()
    return x, y, columns_names


def scale_data_to_range_0_1(x: pd.DataFrame, feature_names: [str], percentage_feature_names: [str]) -> pd.DataFrame:
    for column in feature_names:
        if column in percentage_feature_names:
            x[column] = x[column] / 100.0
        else:
            x[column] = x[column] / x[column].max()
    x = x.fillna(0.0)
    return x


if __name__ == '__main__':
    # Load the data for the experiment
    args = parser.parse_args()
    query_path = os.path.join(args.data_dir, 'query.sql')
    features_path = os.path.join(args.data_dir, 'features.txt')
    target_path = os.path.join(args.data_dir, 'target.txt')

    assert os.path.isfile(query_path), "No json configuration file found at {}".format(query_path)
    assert os.path.isfile(features_path), "No json configuration file found at {}".format(features_path)

    # Set the logger
    set_logger(os.path.join(args.data_dir, 'build_dataset.log'))

    # Load the dataset
    logging.info("Downloading dataset...")
    query = open(query_path, mode='r').read()
    features = [feature.rstrip() for feature in open(features_path, mode='r').readlines()]
    target = open(target_path, mode='r').read()
    x, y = download_data(query=query, database_name=args.database_name, features=features, target=target)
    logging.info("Downloaded dataset. Features shape {}. Target shape {}.".format(x.shape, y.shape))

    # OneHot encoding
    x, y, features = normalize_datatypes(x, y)
    logging.info("Features normalized. Features shape {}. Target shape {}.".format(x.shape, y.shape))

    # Apply feature selection if needed.
    if len(features) > 40:
        logging.info(f"The dataset contains more than 40 features ({len(features)}), we will apply feature selection")
        selected_features = select_features(x,y)
        x = x[selected_features]

    x['user_class'] = y
    out_file_path = os.path.join(args.data_dir, args.out_file_name)
    x.to_csv(out_file_path, index=False)
    logging.info("Finished.")

