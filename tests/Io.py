import os
import unittest
import pandas as pd
import numpy as np
import math
import pickle
from pandas.errors import EmptyDataError
from pandas import Timestamp

from src.movekit.feature_extraction import *
from src.movekit.io import parse_csv, parse_excel



class Test_IO(unittest.TestCase):
    def test_read_csv_file(self):
        ref = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3},
                            'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607,
                                          8: 811, 9: 905, 10: 312},
                            'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01,
                                  7: 390.25, 8: 445.48, 9: 365.86, 10: 405.31},
                            'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82,
                                  7: 405.89, 8: 412.26, 9: 451.76, 10: 417.07}})
        path = '../tests/data/fish-5.csv'
        # create Pandas DataFrame-
        df = parse_csv(path, time_format='undefined')
        pd.testing.assert_frame_equal(ref, df.head(11))

    def test_read_csv_time(self):
        path = '../tests/data/fish-5_time.csv'

        ref = pd.DataFrame({'time': {0: Timestamp('2020-03-27 11:57:07'), 1: Timestamp('2020-03-27 11:57:09'),
                                     2: Timestamp(
            '2020-03-27 11:57:11'), 3: Timestamp('2020-03-27 11:57:13'), 4: Timestamp('2020-03-27 11:57:15'),
                        5: Timestamp('2020-03-27 11:57:17'), 6: Timestamp('2020-03-27 11:57:19'),
                        7: Timestamp('2020-03-27 11:57:21'), 8: Timestamp('2020-03-27 11:57:23'),
                        9: Timestamp('2020-03-27 11:57:25'), 10: Timestamp('2020-03-27 11:57:27')},
               'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811, 9: 905, 10: 905},
               'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01, 7: 390.25, 8: 445.48,
                     9: 365.86, 10: 365.7},
               'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82, 7: 405.89, 8: 412.26,
                     9: 451.76, 10: 451.76}})


        df = parse_csv(path, time_format='undefined')
        pd.testing.assert_frame_equal(ref, df.head(11))

    def test_read_excel_time(self):

        ref = pd.DataFrame({'time': {0: Timestamp('2020-03-27 11:57:07'), 1: Timestamp('2020-03-27 11:57:09'),
                                     2: Timestamp(
            '2020-03-27 11:57:11'), 3: Timestamp('2020-03-27 11:57:13'), 4: Timestamp('2020-03-27 11:57:15'),
                        5: Timestamp('2020-03-27 11:57:17'), 6: Timestamp('2020-03-27 11:57:19'),
                        7: Timestamp('2020-03-27 11:57:21'), 8: Timestamp('2020-03-27 11:57:23'),
                        9: Timestamp('2020-03-27 11:57:25'), 10: Timestamp('2020-03-27 11:57:27')},
               'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811, 9: 905, 10: 905},
               'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01, 7: 390.25, 8: 445.48,
                     9: 365.86, 10: 365.7},
               'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82, 7: 405.89, 8: 412.26,
                     9: 451.76, 10: 451.76}})
        path = '../tests/data/fish-5_time.xlsx'

        df = parse_excel(path, sheet=0, time_format='undefined')
        pd.testing.assert_frame_equal(ref, df.head(11))

    def test_read_csv_file_without_suffix(self):
        # Specify file location WITHOUT file extension-
        path = '../tests/data/fish-5'

        df = parse_csv(path, time_format='undefined')
        self.assertEqual(df.shape[0], 50)

    def test_read_csv_file_empty(self):
        path = '../tests/data/empty.csv'
        self.assertRaises(EmptyDataError, lambda: parse_csv(path, time_format='undefined'))

    def test_read_csv_file_missing_header(self):
        path = '../tests/data/missing-header.csv'
        self.assertRaises(ValueError, lambda: parse_csv(path, time_format='undefined'))

    '''
    Unit test for Microsoft Excel file
    '''

    def test_read_excel_file(self):
        ref = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3},
                            'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607,
                                          8: 811, 9: 905, 10: 312},
                            'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01,
                                  7: 390.25, 8: 445.48, 9: 365.86, 10: 405.31},
                            'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82,
                                  7: 405.89, 8: 412.26, 9: 451.76, 10: 417.07}})
        path = '../tests/data/fish-5.xlsx'

        df = parse_excel(path, sheet=0, time_format='undefined')
        pd.testing.assert_frame_equal(ref, df.head(11))

    def test_read_excel_file_without_suffix(self):
        path = '../tests/data/fish-5'

        df = parse_excel(path, sheet=0, time_format='undefined')
        self.assertEqual(len(df.index), 50)

    def test_read_excel_file_empty(self):
        path = '../tests/data/empty.xlsx'

        self.assertRaises(EmptyDataError, lambda: parse_excel(path, sheet=0, time_format='undefined'))

    def test_read_excel_file_missing_header(self):
        path = '../tests/data/missing-header.xlsx'
        self.assertRaises(ValueError, lambda: parse_excel(path, sheet=0, time_format='undefined'))

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

        df1 = parse_csv(csv_path, time_format='undefined')
        df2 = parse_excel(excel_path, sheet=0, time_format='undefined')

        df3 = parse_csv(csv_time_path, time_format='undefined')
        df4 = parse_excel(excel_time_path, sheet=0, time_format='undefined')

        pd.testing.assert_frame_equal(df1, df2)
        pd.testing.assert_frame_equal(df3, df4)
        #self.assertEqual(df1.all(), df2.all())


if __name__ == '__main__':
    unittest.main()
