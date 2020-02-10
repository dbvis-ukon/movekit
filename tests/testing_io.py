

import os
import unittest
import pandas as pd

# from pandas.util.testing import assert_frame_equal
from pandas.testing import assert_frame_equal

# from pandas.io.common import EmptyDataError
from pandas.errors import EmptyDataError


os.chdir("../src/")


import movekit as mkit
import movekit

# from movekit.io_combined import read_data
from movekit.io_combined import parse_csv
from movekit.io_combined import parse_excel
# from movekit.preprocessing_combined import preprocessing_methods
# from movekit.feature_extraction_combined import feature_extraction_methods
# from movekit.plot import plotting_methods


# Enter absolute/complete path to CSV file-
file_loc = "../examples/datasets/fish-5.csv"
file_loc_xl = "../examples/datasets/fish-5"


# data = pd.read_csv(file_loc)

# data.shape
# (5000, 4)



class Test_IO(unittest.TestCase):


	'''
	Unit tests for CSV file
	'''

	def test_read_csv_file(self):
		path = '../tests/data/fish-5.csv'

		# create Pandas DataFrame-
		df = parse_csv(path)
		self.assertEqual(df.shape[0], 50)


	def test_read_csv_file_without_suffix(self):
		# Specify file location WITHOUT file extension-
		path = '../tests/data/fish-5'

		df = parse_csv(path)
		self.assertEqual(df.shape[0], 50)


	def test_read_csv_file_empty(self):
		path = '../tests/data/empty.csv'

		df = parse_csv(path)
		self.assertRaises(EmptyDataError, df)


	
	def test_read_csv_file_missing_header(self):
		path = '../tests/data/missing-header.csv'

		# df = parse_csv(path)
		df = pd.read_csv(path)

		expected = ['time', 'animal_id', 'x', 'y']
		result = list(df.columns)

		# self.assertCountEqual(result, expected)
		self.assertEqual(result, expected)
	


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
		path = 'empty.xlsx'
		
		self.assertRaises(EmptyDataError, parse_excel(path))


	
	def test_read_excel_file_missing_header(self):
		path = '../tests/data/missing-col.xlsx'
		
		df = pd.read_excel(path)
		expected = ['time', 'animal_id', 'x', 'y']
		result = list(df.columns)

		# self.assertCountEqual(result, expected)
		self.assertEqual(result, expected)
	


	def test_csv_excel_equal(self):
		'''
		Function to compare two files of file types-
		CSV and MS Excel
		'''
		csv_path = '../tests/data/fish-5.csv'
		excel_path = '../tests/data/fish-5.xlsx'

		df1 = parse_csv(csv_path)
		df2 = parse_excel(excel_path)

		pd.testing.assert_frame_equal(df1, df2)
		# self.assertEqual(df1.all(), df2.all())








if __name__ == '__main__':
	unittest.main()


