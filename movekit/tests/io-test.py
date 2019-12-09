import os

import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from pandas.io.common import EmptyDataError

from movekit.io.csv import parse_csv
from movekit.io.excel import parse_excel


def datafile(fn):
    return os.path.join(os.path.dirname(__file__), 'data', fn)


class TestIO(unittest.TestCase):

    """
    Tests loading of different types of csv and excel files
    """

    def test_read_csv_file(self):
        path = 'fish-5.csv'
        df = parse_csv(datafile(path))
        self.assertEqual(len(df.index), 50)

    def test_read_csv_file_without_suffix(self):
        path = 'fish-5'
        df = parse_csv(datafile(path))
        self.assertEqual(len(df.index), 50)

    def test_read_csv_file_empty(self):
        path = 'empty.csv'
        self.assertRaises(EmptyDataError, parse_csv(datafile(path)))

    def test_read_csv_file_missing_header(self):
        path = 'missing-header.csv'
        self.assertRaises(EmptyDataError, parse_csv(datafile(path)))

    def test_read_csv_file_missing_col(self):
        path = 'missing-col.csv'
        self.assertRaises(EmptyDataError, parse_csv(datafile(path)))

    def test_read_excel_file(self):
        path = 'fish-5.xlsx'
        df = parse_excel(datafile(path))
        self.assertEqual(len(df.index), 50)

    def test_read_excel_file_without_suffix(self):
        path = 'fish-5'
        df = parse_excel(datafile(path))
        self.assertEqual(len(df.index), 50)

    def test_read_excel_file_empty(self):
        path = 'empty.xlsx'
        self.assertRaises(EmptyDataError, parse_excel(datafile(path)))

    def test_read_excel_file_missing_header(self):
        path = 'missing-header.xlsx'
        self.assertRaises(EmptyDataError, parse_excel(datafile(path)))

    def test_read_excel_file_missing_col(self):
        path = 'missing-col.xlsx'
        self.assertRaises(EmptyDataError, parse_excel(datafile(path)))

    def test_csv_excel_equal(self):
        csv_path = 'fish-5.csv'
        df1 = parse_csv(datafile(csv_path))
        excel_path = 'fish-5.xlsx'
        df2 = parse_excel(datafile(excel_path))
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == '__main__':
    unittest.main()
