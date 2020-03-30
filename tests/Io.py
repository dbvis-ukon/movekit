

import os
import unittest
import pandas as pd
import numpy as np
import math
import pickle
from pandas.errors import EmptyDataError

from src.movekit.feature_extraction import *



from src.movekit.io import parse_csv, parse_excel



class Test_IO(unittest.TestCase):
    def test_read_csv_file(self):
        path = '../tests/data/fish-5.csv'
        # create Pandas DataFrame-
        df = parse_csv(path)
        self.assertEqual(df.shape[0], 50)

    def test_read_csv_time(self):
        path = '../tests/data/fish-5_time.csv'

        df = parse_csv(path)
        self.assertEqual(df.shape[0], 50)

    def test_read_excel_time(self):
        path = '../tests/data/fish-5_time.xlsx'

        df = parse_excel(path)
        self.assertEqual(df.shape[0], 50)

    def test_read_csv_file_without_suffix(self):
        # Specify file location WITHOUT file extension-
        path = '../tests/data/fish-5'

        df = parse_csv(path)
        self.assertEqual(df.shape[0], 50)

    def test_read_csv_file_empty(self):
        path = '../tests/data/empty.csv'
        self.assertRaises(EmptyDataError, parse_excel(path))

    def test_read_csv_file_missing_header(self):
        path = '../tests/data/missing-col.csv'

        # df = parse_csv(path)
        df = pd.read_csv(path)

        expected = ['time', 'animal_id', 'x', 'y']
        result = list(df.columns)

        # self.assertCountEqual(result, expected)
        self.assertNotEqual(result, expected)

    '''
    Unit test for Microsoft Excel file
    '''

    def test_read_excel_file(self):
        path = '../tests/data/fish-5.xlsx'

        df = parse_excel(path)
        self.assertEqual(df.shape[0], 50)

    def test_read_excel_file_without_suffix(self):
        path = '../tests/data/fish-5'

        df = parse_excel(path)
        self.assertEqual(len(df.index), 50)

    def test_read_excel_file_empty(self):
        path = '../tests/data/empty.xlsx'

        self.assertRaises(EmptyDataError, parse_excel(path))

    def test_read_excel_file_missing_header(self):
        path = '../tests/data/missing-col.xlsx'

        df = pd.read_excel(path)
        expected = ['time', 'animal_id', 'x', 'y']
        result = list(df.columns)

        # self.assertCountEqual(result, expected)
        self.assertNotEqual(expected, result)

    def test_csv_excel_equal(self):
        '''
        Function to compare two files of file types-
        CSV and MS Excel
        '''

        # Parsing with numeric time stamps
        csv_path = '../tests/data/fish-5.csv'
        excel_path = '../tests/data/fish-5.xlsx'

        # Parsing with strings to datetime stamps
        csv_time_path = '../tests/data/fish-5_time.csv'
        excel_time_path = '../tests/data/fish-5_time.xlsx'

        df1 = parse_csv(csv_path)
        df2 = parse_excel(excel_path)

        df3 = parse_csv(csv_time_path)
        df4 = parse_excel(excel_time_path)

        pd.testing.assert_frame_equal(df1, df2)
        pd.testing.assert_frame_equal(df3, df4)
        #self.assertEqual(df1.all(), df2.all())


if __name__ == '__main__':
    unittest.main()
