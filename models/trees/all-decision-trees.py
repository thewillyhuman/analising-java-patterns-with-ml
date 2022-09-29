# -*-encoding:utf-8-*-


import os
from typing import List

import graphviz
import matplotlib.pyplot as plt
import pandas as pd
from dtreeviz.trees import dtreeviz
from sklearn import tree

from util import load_file, show_accuracies, \
    cross_validation

TREE_FIGURES_DIR = "tree/"
CSVS_DIR = "../../data/unique/"
CSV_FILE_NAMES = ["programs/dataset_and_target.csv", "type_defs/dataset_and_target.csv",
                  "field_defs/dataset_and_target.csv",
                  "method_defs/dataset_and_target.csv", "expressions/dataset_and_target.csv",
                  "statements/dataset_and_target.csv", "types/dataset_and_target.csv",
                  "het1/dataset_and_target.csv", "het2/dataset_and_target.csv",
                  "het3/dataset_and_target.csv", "het4/dataset_and_target.csv",
                  "het5/dataset_and_target.csv", ]

MIN_TREE_DEPTH = 1
MAX_TREE_DEPTH = 3
XV_FOLDS = 10

SHOW_TABLES = False
SHOW_ACCURACIES = True
SHOW_RULES = True
CREATE_UNLIMITED_TREE = False

TARGET_FEATURE = "user_class"
CLASS_NAMES = ["low", "high"]


def render_tree_plt(clf, feature_manes: List[str], class_names: List[str], file_name: str):
    fig = plt.figure(figsize=(30, 30))
    tree.plot_tree(clf,
                   feature_names=feature_manes,
                   class_names=class_names,
                   filled=True)
    fig.savefig(TREE_FIGURES_DIR + file_name)


def render_tree_graphviz(clf, feature_manes: List[str], class_names: List[str], file: str):
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_manes,
                                    class_names=class_names,
                                    filled=True)
    file_name, file_extension = os.path.splitext(file)
    graph = graphviz.Source(dot_data, format=file_extension.replace(".", ""))
    graph.render(TREE_FIGURES_DIR + file_name)


def render_tree_dtreeviz(clf, feature_manes: List[str], class_names: List[str], target_name: str,
                         file_name: str):
    viz = dtreeviz(clf, X.values, y.values.reshape(-1),
                   target_name=target_name,
                   class_names=class_names,
                   feature_names=feature_manes)
    viz.save(TREE_FIGURES_DIR + file_name)


def show_rules(clf, feature_names: List[str]):
    if SHOW_RULES:
        text_representation = tree.export_text(clf, feature_names=feature_names)
        print(text_representation)


def show_trees(clf, feature_names: List[str], class_names: List[str], target_feature: str,
               file_name_prefix: str):
    if not os.path.exists(TREE_FIGURES_DIR):
        os.makedirs(TREE_FIGURES_DIR)
    # render_tree_plt(clf, list(feature_names), class_names, f"{file_name_prefix}-plt.png")
    # render_tree_graphviz(clf, list(feature_names), class_names, f"{file_name_prefix}-graphviz.png")
    # render_tree_dtreeviz(clf, list(feature_names), class_names, target_feature,
    # f"{file_name_prefix}-dtreeviz.svg")


if __name__ == "__main__":

    for csv_file_name in CSV_FILE_NAMES:
        file_name = os.path.splitext(csv_file_name)[0]  # file name without the extension
        print("\n", "-" * 10, f"Processing '{file_name}' file", "-" * 10, "\n")
        # load file
        print("loading data...")
        df, X, y = load_file(CSVS_DIR + csv_file_name, None, TARGET_FEATURE, SHOW_TABLES)
        X = pd.get_dummies(X)
        X = X.astype('float32')
        X = X.fillna(0.0)
        y = y.squeeze()
        y = y.apply(lambda value: 0 if value == "low" else 1)  # high will be 1 and low will be 0.
        y = y.astype('float32')
        feature_names = list(X.columns)

        for max_depth in range(MIN_TREE_DEPTH, MAX_TREE_DEPTH + 1):
            print(f"Depth: {max_depth}...")
            # train a decision tree
            clf = tree.DecisionTreeClassifier(max_depth=max_depth)
            fitted_clf = clf.fit(X, y)
            # show the values
            show_accuracies(fitted_clf, df, X, y, TARGET_FEATURE, SHOW_ACCURACIES)
            cross_validation(clf, X, y, SHOW_ACCURACIES, XV_FOLDS, SHOW_TABLES)
            show_rules(fitted_clf, feature_names)
            show_trees(fitted_clf, feature_names, CLASS_NAMES, TARGET_FEATURE,
                       f"tree-{os.path.splitext(csv_file_name)[0]}-{max_depth}")

        if CREATE_UNLIMITED_TREE:
            print(f"No limited depth")
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(X, y)
            # show the values
            show_accuracies(clf, df, X, y, TARGET_FEATURE, SHOW_ACCURACIES)
            cross_validation(clf, X, y, SHOW_ACCURACIES, 10, SHOW_TABLES)
            show_rules(clf, feature_names)
            # show_trees(feature_names, CLASS_NAMES, TARGET_FEATURE, f"tree-{file_name}-no-limit")
