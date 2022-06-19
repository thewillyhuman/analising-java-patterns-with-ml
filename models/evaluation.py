from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def evaluate_logistic_regression(log_reg: LogisticRegression, x_test, y_test) -> (
        float, float, float, float):
    """Evaluate the model

    Evaluates the model by means of accuracy, recall, precision and f1 scores.

    :param log_reg: is the model to evaluate, needs to be fitted.
    :param x_test: is the test split of the features.
    :param y_test: is the test split of the target
    :return: accuracy, recall, precision and f1 scores
    """
    y_pred = log_reg.predict(x_test)
    return accuracy_score(y_test, y_pred, ), \
           recall_score(y_test, y_pred, average='micro'), \
           precision_score(y_test, y_pred, average='micro'), \
           f1_score(y_test, y_pred, average='micro')
