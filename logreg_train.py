import argparse
import logging
import os

from build_dataset import normalize_datatypes, scale_data_to_range_0_1
from models.logistic_regression import build_model, train_and_evaluate, save_coefs_to_excel
from models.utils import Params, set_logger, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--model-dir', type=str, required=True)
parser.add_argument('--database-name', type=str, default='patternminingV2')

if __name__ == '__main__':

    # Load the data for the experiment
    args = parser.parse_args()
    dataset = Dataset(data_dir=args.data_dir)
    model_path = os.path.join(args.model_dir, 'model.joblib')

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Load the parameters from the experiment params.json file in model_dir
    json_path = os.path.join(args.model_dir, 'params.json')
    if os.path.isfile(json_path):
        params = Params(json_path)
        logging.info(f"Parameters file found at [{json_path}].")
    else:
        # Write parameters in json file
        params = None
        logging.info("Parameters file not found.")

    # Load the dataset
    x, y = dataset.x, dataset.y
    logging.info(f"Dataset loaded. Shape: {x.shape}, {y.shape}.")

    # OneHot encoding
    x, y, features = normalize_datatypes(x, y)
    logging.info(f"OneHot applied. Shape after: {x.shape}, {y.shape}.")

    # Scale features to range 0 1
    x = scale_data_to_range_0_1(x, features, dataset.percentage_features)
    logging.info(f"Scaler applied. Shape after: {x.shape}, {y.shape}.")

    # Create the model
    logging.info("Creating model for params [{}]...".format(params))
    elnet_model = build_model(params=params)

    # Train the model
    logging.info("Starting training")
    model, acc, rec, pre, f1 = train_and_evaluate(log_reg=elnet_model, x=x, y=y)
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
    params.weights = dict(zip(features, elnet_model.coef_.reshape(-1).tolist()))

    # Write parameters in json file
    json_path = os.path.join(args.model_dir, 'metrics_eval_best_weights.json')
    params.save(json_path)
    logging.info("Data saved.")

    # Save results to excel
    out_file_path = os.path.join(args.model_dir, 'elnet_coefs.xlsx')
    save_coefs_to_excel(log_reg=elnet_model, features=features, out_file_path=out_file_path)
    logging.info("All saved and finished.")
