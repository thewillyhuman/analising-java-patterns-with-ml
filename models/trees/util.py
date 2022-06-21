# -*-encoding:utf-8-*-

import os
import time
from fnmatch import fnmatch
import sys

from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from dtreeviz.trees import dtreeviz
import os
from typing import List, Set, Any, Union, Optional


def calcular_accuracy_major_group_classifier(df: pd.DataFrame, field_name: str) -> float:
    counts = df[field_name].value_counts(normalize=True)
    return max(counts[0], counts[1])


def convert_to_one_hot(df: pd.DataFrame, feature_names_to_one_hot_encode: Set[str], show_tables: bool) -> pd.DataFrame:
    one_hot = pd.get_dummies(df, columns=feature_names_to_one_hot_encode)  # generate one-hot columns
    if show_tables:
        print(one_hot.describe())
        print(one_hot.shape)
        print(one_hot.head())
    return one_hot


def convert_boolean_features(df: pd.DataFrame, boolean_feature_names: Set[str], show_tables: bool) -> pd.DataFrame:
    for boolean_feature_name in boolean_feature_names:
        df[boolean_feature_name].replace({False: 0, True: 1}, inplace=True)
    if show_tables:
        print(df.describe())
        print(df.shape)
        print(df.head())
    return df


def load_file(file_name: str, feature_names: Optional[Set[str]], target_feature: str, show_tables: bool):
    df = pd.read_csv(file_name)
    if not feature_names:  # if no feature names are specified, they will be all but the target
        X = df.loc[:, df.columns != target_feature]
    else:
        X = df.loc[:, feature_names]
    y = df.loc[:, [target_feature]]
    if show_tables:
        print(df.describe())
        print(df.shape)
    if show_tables:
        pd.options.display.max_columns = 2000
        print(X.describe())
        print(X.shape)
        print(y.describe())
        print(y.shape)
    return df, X, y


def show_accuracies(clf, df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, target_feature: str, show_accuracies: bool):
    if show_accuracies:
        print("Accuracy with all the data: {0:.4f}.".format(accuracy_score(y, clf.predict(X))))
        print("Accuracy de la clase mayoritaria {0:.4f}.".format(
            calcular_accuracy_major_group_classifier(df, target_feature)))


def cross_validation(clf, X: pd.DataFrame, y: pd.DataFrame, show_accuracies: bool, xv_folds: int, show_tables: bool):
    if show_accuracies:
        scores = cross_val_score(clf, X, y, cv=xv_folds)
        print("Scores for cross validation: ", end='')
        print(scores)
        print("XV: {:.4f} accuracy with a standard deviation of {:.4f}.".format(scores.mean(), scores.std()))


