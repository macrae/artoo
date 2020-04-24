import numpy as np
from hypothesis import strategies
from hypothesis.strategies._internal.lazy import LazyStrategy
from typing import NamedTuple
from tabular_data import TabularData, tabular_parser


def test_tabular_parser():
    tabular_data = tabular_parser("./data/female_names_top100_2019.csv")
    assert isinstance(tabular_data, TabularData)
    assert isinstance(tabular_data.column_names, np.ndarray)
    assert isinstance(tabular_data.data, np.ndarray)
    assert len(tabular_data.get_column("Amount")) == 100


def test_create_column_strategy():
    tabular_data = tabular_parser("./data/female_names_top100_2019.csv")
    some_amount = tabular_data.create_column_strategy("Amount")
    assert isinstance(some_amount, LazyStrategy)


def test_create_record_strategy():
    tabular_data = tabular_parser("./data/female_names_top100_2019.csv")
    some_record = tabular_data.create_record_strategy()
    assert isinstance(some_record.Amount, LazyStrategy)