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
from typing import List, Set, Any, Union

CSV_FILE_NAME = "csvs/type_definitions_unique.csv"
BOOLEAN_FEATURE_NAMES: Set[str] = {"type_def__visibility", 'type_def__is_final', "type_def__is_extension",
                            "type_def__is_in_default_package",
                            "type_def__is_abstract",
                           "type_def__is_nested", "type_def__is_static",
                           }
NOMINAL_FEATURE_NAMES: Set[str] = {"type_def__syntactic_category",
                            "type_def__naming_convention",
                           }
NUMERIC_FEATURE_NAMES: Set[str] = { "type_def__number_of_annotations",
                           "type_def__number_of_implements",
                           "type_def__number_of_generics", "type_def__number_of_methods",
                           "type_def__percentage_overloaded_methods",
                           "type_def__number_of_constructors", "type_def__number_of_fields",
                           "type_def__number_of_nested_types",
                           "type_def__number_of_inner_types",
                           "type_def__number_of_static_nested_types",
                           "type_def__percentage_of_static_fields", "type_def__percentage_of_static_methods",
                           "type_def__number_of_static_blocks",
                           }
TARGET_FEATURE = "user_class"
CLASS_NAMES = ["low", "high"]


def get_all_the_feature_names():
    return BOOLEAN_FEATURE_NAMES.union(NOMINAL_FEATURE_NAMES).union(NUMERIC_FEATURE_NAMES)
