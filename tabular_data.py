import collections
from csv import reader
from functools import reduce
from typing import NamedTuple

import hypothesis.strategies as st
import numpy as np
from numpy import array


class TabularData(NamedTuple):
    column_names: array
    data: array

    def get_column(self, column_name: str) -> np.ndarray:
        return self.data[:, np.where(self.column_names == column_name)].reshape(-1)

    def create_column_strategy(self, column_name: str):
        column = self.get_column(column_name)
        if all(isinstance(x, bool) for x in column):
            return st.sampled_from([True, False])
        elif all(isinstance(x, int) for x in column):
            return st.one_of(st.integers())
        elif all(isinstance(x, float) for x in column):
            return st.one_of(st.floats())
        elif all(isinstance(x, str) for x in column):
            return st.sampled_from(column)
        else:
            return st.nothing()

    def create_record_strategy(self):
        field_names = [x.replace(" ", "_") for x in self.column_names]
        some_record = collections.namedtuple("some_record", field_names)
        return some_record(
            *[self.create_column_strategy(field_name) for field_name in field_names]
        )


def tabular_parser(path: str, header: bool = True) -> TabularData:
    """Given a path to a csv file, parse it as a list.

    Parameters
    ----------
    path : str
        path to file

    Returns
    -------
    TabularData
        csv file parsed into TabularData
    """
    with open(path, "r") as read_obj:
        csv_reader = reader(read_obj)
        list_of_rows = list(csv_reader)
    rows = np.array(list_of_rows)

    if header:
        return TabularData(column_names=rows[0, :], data=rows[1:, :])
    else:
        return TabularData(column_names=None, data=rows[1:, :])
