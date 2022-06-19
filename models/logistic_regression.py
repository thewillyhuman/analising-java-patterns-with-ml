"""Utility functions to work with scikit-learn Logistic Regression model"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from models.evaluation import evaluate_model
from models.utils import Params


def build_model(params: Params = None, n_jobs: int = -1) -> LogisticRegression:
    """Builds a logistic regression with elasticnet penalty with default or given params."""
    return LogisticRegression(solver='saga', penalty='elasticnet', C=params.C,
                              l1_ratio=params.l1_ratio, n_jobs=n_jobs, max_iter=1000)


def train_and_evaluate(log_reg: LogisticRegression, x: pd.DataFrame, y: pd.Series) -> (
        LogisticRegression, float, float, float, float):
    """Fits and evaluates the given logistic regression model

    :param log_reg: is the model to fit. Needs to be parametrized.
    :param x: is the features dataset.
    :param y: is the targets dataset.
    :return: the fitted model, the accuracy, the recall, the precision and the f1 score.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)
    log_reg.fit(x_train, y_train)
    acc, rec, pre, f1 = evaluate_model(model=log_reg, x_test=x_test, y_test=y_test)

    return log_reg, acc, rec, pre, f1


def save_coefs_to_excel(log_reg: LogisticRegression, features: [str], out_file_path: str) -> None:
    """Saves the coefficients """
    tmp_df = pd.DataFrame(log_reg.coef_).T
    tmp_df['features'] = features
    tmp_df.to_excel(excel_writer=out_file_path)
