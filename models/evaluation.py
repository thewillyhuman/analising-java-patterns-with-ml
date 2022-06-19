from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def evaluate_model(model, x_test, y_test) -> (float, float, float, float):
    """Evaluates a model

    Evaluates the model by means of accuracy, recall, precision and f1 scores.

    :param model: is the model to evaluate, needs to be fitted.
    :param x_test: is the test split of the features.
    :param y_test: is the test split of the target
    :return: accuracy, recall, precision and f1 scores
    """
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred, ), recall_score(y_test, y_pred), precision_score(
        y_test, y_pred), f1_score(y_test, y_pred)
