from typing import List, Set, Any, Union
import pandas as pd

rules_str = "[[type_def__is_in_default_package=0] V [type_def__percentage_of_static_fields=<20.0^type_def__visibility=0] V [type_def__is_extension=1] V [type_def__percentage_of_static_methods=>33.33] V [type_def__percentage_of_static_fields=<20.0^type_def__percentage_of_static_methods=4.35-33.33] V [type_def__number_of_implements=1] V [type_def__number_of_constructors=<1.0^type_def__number_of_implements=2] V [type_def__syntactic_category_Class=0] V [type_def__is_nested=1]]"


def get_rules_from_str(rules_str: str) -> List[str]:
    """Returns all the rules (disjunctions) from a string"""
    original_rules = rules_str.split(" V ")
    processed_rules = list()
    for rule in original_rules:
        rule2 = rule.replace('[', "").replace("]", "")  # removes [  and ]
        rule3 = rule2.strip()  # removes blanks
        processed_rules.append(rule3)
    return processed_rules


class Conjunction:
    pass


class Conjunction:

    def __init__(self, feature_name: str, operator: str, value: str):
        self.feature_name = feature_name
        self.operator = operator
        self.value = value

    __original_operators = ['=>', '=<', '=']
    __target_operators = ['>=', '<=', '==']

    def __str__(self):
        return f"{self.feature_name} {self.operator} {self.value}"

    def __repr__(self):
        return str(self)

    @classmethod
    def create_conjunction_from_str(cls, conjunction_str: str) -> List[Conjunction]:
        feature_name, operator, value = Conjunction.__split_conjunction(conjunction_str)
        if operator == '==' and "-" in value:  # it is a range conjunction: feature = 3-4
            return Conjunction.__process_range_conjunction(feature_name, value)
        return [Conjunction(feature_name, operator, value)]

    @classmethod
    def __split_conjunction(cls, conjunction_str: str) -> (str, str, str):
        for index in range(len(Conjunction.__original_operators)):
            operator = Conjunction.__original_operators[index]
            operands = conjunction_str.split(operator)
            if len(operands) == 2:
                return operands[0], Conjunction.__target_operators[index], operands[1]
        assert False, f"The conjunction '{conjunction_str}' is not well defined."

    @classmethod
    def __process_range_conjunction(cls, feature_name: str, value_str: str) -> List[Conjunction]:
        """ For conjunctions like feature=4.35-33.33 which really are feature>=4.35 and feature<=33.33"""
        values = value_str.split('-')
        assert len(values) == 2
        return [Conjunction(feature_name, '>=', values[0]), Conjunction(feature_name, '<=', values[1])]

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takes a dataframe and filters it with the corresponding conjunction"""
        return df.query(str(self))


def __get_conjunctions_from_rule(rule: str) -> List[Conjunction]:
    """Given a rule as a string, returns the rule as a list of conjunctions"""
    conjunctions_str = rule.split("^")
    all_conjunctions = list()
    for conjunction in conjunctions_str:
        conjunction2 = conjunction.strip()  # removes blanks
        conjunctions = Conjunction.create_conjunction_from_str(conjunction2)
        all_conjunctions.extend(conjunctions)
    return all_conjunctions


def filter_dataframe_from_rule(df: pd.DataFrame, rule_str: str) -> pd.DataFrame:
    conjunctions = __get_conjunctions_from_rule(rule_str)
    for conjunction in conjunctions:
        df = conjunction.filter_dataframe(df)
    return df

