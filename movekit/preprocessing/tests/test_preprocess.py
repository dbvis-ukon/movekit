"""
Ensure that we can use all relevant IO functions
"""
import os

import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from pandas.io.common import EmptyDataError
import numpy as np

from movekit.preprocessing.preprocess import clean


class TestPreprocess(unittest.TestCase):

    def test_clean(self):
        clean_data = {
            'animal_id': [1, 2, 1, 2],
            'time': [1, 1, 2, 2],
            'x': [10, 19, 11.0, 14.1],
            'y': [5, 20, 4, 14.2]}
        df_clean = pd.DataFrame(clean_data)

        err_data = df_clean.append(
            {'animal_id': np.nan, 'time': np.nan, 'x': np.nan, 'y': np.nan, }, ignore_index=True)
        assert_frame_equal(df_clean, clean(pd.DataFrame(err_data)))

        err_data = df_clean.append(
            {'animal_id': 1, 'time': np.nan, 'x': 10, 'y': 11, }, ignore_index=True)
        assert_frame_equal(df_clean, clean(pd.DataFrame(err_data)))

        err_data = df_clean.append(
            {'animal_id': np.nan, 'time': 10, 'x': 10, 'y': 11, }, ignore_index=True)
        assert_frame_equal(df_clean, clean(pd.DataFrame(err_data)))

        err_data = df_clean.append(
            {'animal_id': 1, 'time': 1, 'x': 10, 'y': 5, }, ignore_index=True)
        assert_frame_equal(df_clean, clean(pd.DataFrame(err_data)))


if __name__ == '__main__':
    unittest.main()
