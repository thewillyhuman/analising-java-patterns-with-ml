# -*-encoding:utf-8-*-


from sklearn import tree
import matplotlib.pyplot as plt
import graphviz
from dtreeviz.trees import dtreeviz
import os
from typing import List, Set, Any, Union
import pandas as pd
from configuration import get_all_the_feature_names
from util import load_file, convert_to_one_hot, convert_boolean_features, show_accuracies, cross_validation
import wittgenstein as lw
from rule_parsing import get_rules_from_str, filter_dataframe_from_rule

MIN_CONFIDENCE = 0.9
MIN_SUPPORT = 5/100
MAX_NUMBER_CONJUNCTIONS = 3

SHOW_TABLES = False
SHOW_ACCURACIES = True

CSV_FILE_NAMES = ["expressions_unique.csv", "field_defs_unique.csv", "method_defs_unique.csv", "programs_unique.csv",
                  "statements_unique.csv", "type_defs_unique.csv", "types_unique.csv",
                  "het1_unique.csv", "het2_unique.csv", "het3_unique.csv", "het4_unique.csv", "het5_unique.csv"]

TARGET_FEATURE = "user_class"
CLASS_NAMES = ["low", "high"]
CSVS_DIR = "csvs/"



def compute_confidence_and_support(df: pd.DataFrame, rule: str) -> (float, float):
    number_of_instances = df.shape[0]
    filtered_df = filter_dataframe_from_rule(df, rule)
    rule_instances = filtered_df.shape[0]
    rule_instances_high = filtered_df.loc[filtered_df[TARGET_FEATURE] == 1].shape[0]  # those whose expertise is high
    return rule_instances_high / rule_instances, rule_instances / number_of_instances


if __name__ == "__main__":

    for csv_file_name in CSV_FILE_NAMES:
        file_name = os.path.splitext(csv_file_name)[0]  # file name without the extension
        print("\n", "-"*10, f"Processing '{file_name}' file", "-"*10, "\n")
        # load file
        print("loading data...")
        df, X, y = load_file(CSVS_DIR + csv_file_name, None, TARGET_FEATURE, SHOW_TABLES)
        y[TARGET_FEATURE] = y[TARGET_FEATURE].astype(int)
        feature_names = list(X.columns)
        whole_df = pd.concat([X, y], axis=1, join='inner')

        for model in [lw.IREP(), lw.RIPPER()]:
            for high_expertise in [True, False]:
                print(f"\nModel type: {model.__class__.__name__}, High expertise: {high_expertise}...")
                if not high_expertise:
                    y[TARGET_FEATURE].replace({0: 1, 1: 0}, inplace=True)

                model.__init__()
                model.fit(X, y)
                print("{} rules found.".format(str(model.ruleset_).count('V')+1))
                #  print(model.ruleset_)
                show_accuracies(model, df, X, y, TARGET_FEATURE, SHOW_ACCURACIES)
                print()

                rules = get_rules_from_str(str(model.ruleset_))
                for rule in rules:
                    number_of_conjunctions = rule.count('^') + 1
                    confidence, support = compute_confidence_and_support(whole_df, rule)
                    if confidence >= MIN_CONFIDENCE or confidence <= (1-MIN_CONFIDENCE):
                        if support >= MIN_SUPPORT:
                            if number_of_conjunctions <= MAX_NUMBER_CONJUNCTIONS:
                                print(f"Rule: {rule}")
                                print(f"Support: {support * 100:.{2}f}%")
                                print(f"Confidence: {confidence * 100:.{2}f}% (high), {(1 - confidence) * 100:.{2}f}% (low)")
                                print()


