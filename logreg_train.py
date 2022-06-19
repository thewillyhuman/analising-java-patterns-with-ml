import argparse
import logging
import os

from build_dataset import download_data, normalize_datatypes, scale_data_to_range_0_1
from models.models import build_elastic_log_reg_model, train_and_evaluate_log_reg, \
    save_log_reg_coefs_to_excel
from models.utils import Params, set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True,
                    help="Directory containing query and features")
parser.add_argument('--model-dir', type=str, required=True, help="Directory containing params.json")
parser.add_argument('--database-name', type=str, default='patternminingV2')

if __name__ == '__main__':

    # Load the data for the experiment
    args = parser.parse_args()
    query_path = os.path.join(args.data_dir, 'query.sql')
    features_path = os.path.join(args.data_dir, 'features.txt')
    target_path = os.path.join(args.data_dir, 'target.txt')
    percentage_features_path = os.path.join(args.data_dir, 'percentage_features.txt')
    assert os.path.isfile(query_path), "No json configuration file found at {}".format(query_path)
    assert os.path.isfile(features_path), "No json configuration file found at {}".format(
        features_path)
    assert os.path.isfile(
        percentage_features_path), "No json configuration file found at {}".format(
        percentage_features_path)

    # Load the parameters from the experiment params.json file in model_dir
    json_path = os.path.join(args.model_dir, 'params.json')
    if os.path.isfile(json_path):
        params = Params(json_path)
    else:
        params = None

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Load the dataset
    logging.info("Downloading dataset...")
    query = open(query_path, mode='r').read()
    features = [feature.rstrip() for feature in open(features_path, mode='r').readlines()]
    target = open(target_path, mode='r').read()
    percentage_features = [feature.rstrip() for feature in
                           open(percentage_features_path, mode='r').readlines()]
    x, y = download_data(query=query, database_name=args.database_name, features=features,
                         target=target)
    logging.info("Downloaded dataset. Features shape {}. Target shape {}.".format(x.shape, y.shape))

    # OneHot encoding
    x, y, features = normalize_datatypes(x, y)
    logging.info(
        "Features normalized. Features shape {}. Target shape {}.".format(x.shape, y.shape))

    # Scale features to range 0 1
    x = scale_data_to_range_0_1(x, features, percentage_features)
    logging.info("Features scaled. Features shape {}. Target shape {}.".format(x.shape, y.shape))

    # Create the model
    logging.info("Creating model for params [{}]...".format(params))
    elnet_model = build_elastic_log_reg_model(params=params)

    # Train the model
    logging.info("Starting training")
    model, acc, rec, pre, f1 = train_and_evaluate_log_reg(log_reg=elnet_model, x=x, y=y)
    logging.info("Training finished.")

    logging.info(f'Accuracy: {acc}')
    logging.info(f'Recall: {rec}')
    logging.info(f'Precision score: {pre}')
    logging.info(f'F1 score: {f1}')
    logging.info(f'Param: {model}')

    params.accuracy = acc
    params.recall = rec
    params.precision = pre
    params.f1 = f1

    # Store the model betas in the output file.
    for coefs in elnet_model.coef_:
        for (feature, coef) in zip(features, coefs):
            exec(f"params.{feature} = {coef}")

    # Write parameters in json file
    json_path = os.path.join(args.model_dir, 'metrics_eval_best_weights.json')
    params.save(json_path)
    logging.info("Data saved.")

    # Save results to excel
    out_file_path = os.path.join(args.model_dir, 'elnet_coefs.xlsx')
    save_log_reg_coefs_to_excel(log_reg=elnet_model, features=features, out_file_path=out_file_path)
    logging.info("All saved and finished.")
