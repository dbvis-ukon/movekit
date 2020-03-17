

import os
import unittest
import pandas as pd
import numpy as np
import math
import pickle


# from pandas.util.testing import assert_frame_equal
from pandas.testing import assert_frame_equal
import sys
os.chdir("C:/Users/lukas/Dropbox/Movekit/")
from src.movekit.feature_extraction import medoid_computation, grouping_data, compute_average_speed, euclidean_dist, \
	computing_stops, compute_absolute_features, extract_features, ts_feature, ts_all_features, regrouping_data
# from pandas.io.common import EmptyDataError
from pandas.errors import EmptyDataError


#os.chdir("../src/")


#from src.movekit.feature_extraction import grouping_data
from src. movekit.feature_extraction import compute_distance_and_direction
#from src.movekit.feature_extraction import compute_average_speed, euclidean_dist
from src.movekit.feature_extraction import compute_average_acceleration

# from movekit.io_combined import read_data
# from movekit.io_combined import parse_csv
# from movekit.io_combined import parse_excel
# from movekit.preprocessing_combined import preprocessing_methods
# from movekit.feature_extraction_combined import feature_extraction_methods
# from movekit.plot import plotting_methods
os.chdir("C:/Users/lukas/Dropbox/Movekit/tests")
# Miss grouping data
# Miss distance and distraction
# Miss absolute features
# Miss extract features
# Miss computing stops
# Miss medoid computation - but in other file
# Miss distance_euklidean_matrix
# Miss eucidean_dist
# Miss compute_similarity
# Miss ts_all_features
# Miss ts_feature
# Miss explore_features
# Miss explore_features-geospatial

class Test_Feature_Extraction(unittest.TestCase):
	'''
	Unit Tests for Feature Extraction
	'''

	def test_grouping_data(self):
		"""
		Testing grouping data function by animal ID for optimal tracking.
		:return:
		"""
		inp = pd.read_csv("../tests/data/records.csv")
		with open('../tests/data/dict_groups.pkl', 'rb') as handle:
			ref = pickle.load(handle)
		case = grouping_data(inp)
		self.assertTrue((ref[312].all() == case[312].all()).all(), "Results don't match")
		self.assertTrue((ref[511].all() == case[511].all()).all(), "Results don't match")
		self.assertTrue((ref[607].all() == case[607].all()).all(), "Results don't match")
		self.assertTrue((ref[811].all() == case[811].all()).all(), "Results don't match")
		self.assertTrue((ref[905].all() == case[905].all()).all(), "Results don't match")

	def test_regrouping_data(self):
		with open('../tests/data/dict_groups.pkl', 'rb') as handle:
			inp = pickle.load(handle)
		ref = pd.read_csv("../tests/data/regroups.csv")
		case = regrouping_data(inp)
		self.assertEqual((ref.all() == case.all()).all(), True, "Results don't match")

	def test_eucledian_dist(self):
		"""
		Testing euclidean distance calculation. Result of function on records data set is compared with reference over
		all columns.
		Note: This test also includes the functions "compute_similarity()" and "similarity_computation()" since
		"euclidean_dist()" builds upon them.
		:return: Logical, if function returns expected result on given data frames.
		"""
		ref = pd.read_csv("../tests/data/euclidean_dist.csv")
		inp = pd.read_csv("../tests/data/records.csv")
		case = euclidean_dist(inp)
		case.to_csv("../tests/data/euc_records.csv")
		case = case.rename(columns = str)
		self.assertEqual((ref.reset_index(drop=True).all() == case.reset_index(drop=True).all()).all(), True,
						 "Results don't match")


	def test_compute_distance_and_direction(self):
		with open('../tests/data/dist_and_dir.pickle', 'rb') as handle:
			ref = pickle.load(handle)
		with open('../tests/data/dict_groups.pkl', 'rb') as handle:
			inp = pickle.load(handle)
		case = compute_distance_and_direction(inp)
		self.assertTrue((ref[312].all() == case[312].all()).all(), "Results don't match")
		self.assertTrue((ref[511].all() == case[511].all()).all(), "Results don't match")
		self.assertTrue((ref[607].all() == case[607].all()).all(), "Results don't match")
		self.assertTrue((ref[811].all() == case[811].all()).all(), "Results don't match")
		self.assertTrue((ref[905].all() == case[905].all()).all(), "Results don't match")

	def test_compute_average_speed(self):
		with open('../tests/data/av_speed.pickle', 'rb') as handle:
			ref = pickle.load(handle)
		with open('../tests/data/dist_and_dir.pickle', 'rb') as handle:
			inp = pickle.load(handle)
		case = compute_average_speed(inp, fps = 10)
		self.assertTrue((ref[312].all() == case[312].all()).all(), "Results don't match")
		self.assertTrue((ref[511].all() == case[511].all()).all(), "Results don't match")
		self.assertTrue((ref[607].all() == case[607].all()).all(), "Results don't match")
		self.assertTrue((ref[811].all() == case[811].all()).all(), "Results don't match")
		self.assertTrue((ref[905].all() == case[905].all()).all(), "Results don't match")

	def test_average_acceleration(self):
		with open('../tests/data/av_accel.pickle', 'rb') as handle:
			ref = pickle.load(handle)
		with open('../tests/data/av_speed.pickle', 'rb') as handle:
			inp = pickle.load(handle)
		case = compute_average_acceleration(inp, fps = 10)
		self.assertTrue((ref[312].all() == case[312].all()).all(), "Results don't match")
		self.assertTrue((ref[511].all() == case[511].all()).all(), "Results don't match")
		self.assertTrue((ref[607].all() == case[607].all()).all(), "Results don't match")
		self.assertTrue((ref[811].all() == case[811].all()).all(), "Results don't match")
		self.assertTrue((ref[905].all() == case[905].all()).all(), "Results don't match")

	def test_computing_stops(self):
		with open('../tests/data/stops.pickle', 'rb') as handle:
			ref = pickle.load(handle)
		with open('../tests/data/av_accel.pickle', 'rb') as handle:
			inp = pickle.load(handle)
		case = computing_stops(inp, threshold_speed = 0.5)
		self.assertTrue((ref[312].all() == case[312].all()).all(), "Results don't match")
		self.assertTrue((ref[511].all() == case[511].all()).all(), "Results don't match")
		self.assertTrue((ref[607].all() == case[607].all()).all(), "Results don't match")
		self.assertTrue((ref[811].all() == case[811].all()).all(), "Results don't match")
		self.assertTrue((ref[905].all() == case[905].all()).all(), "Results don't match")

	def test_compute_absolute_features(self):
		with open('../tests/data/stops.pickle', 'rb') as handle:
			ref = pickle.load(handle)
		with open('../tests/data/dict_groups.pickle', 'rb') as handle:
			inp = pickle.load(handle)
		case = compute_absolute_features(inp, fps = 10 , stop_threshold = 0.5)
		self.assertTrue((ref[312].all() == case[312].all()).all(), "Results don't match")
		self.assertTrue((ref[511].all() == case[511].all()).all(), "Results don't match")
		self.assertTrue((ref[607].all() == case[607].all()).all(), "Results don't match")
		self.assertTrue((ref[811].all() == case[811].all()).all(), "Results don't match")
		self.assertTrue((ref[905].all() == case[905].all()).all(), "Results don't match")

	def test_extract_features(self):
		ref = pd.read_csv('../tests/data/extracted_features.csv')
		inp = pd.read_csv('../tests/data/records.csv')
		case = extract_features(inp)
		self.assertEqual((ref.all() == case.all()).all(), True, "Results don't match")

	def test_ts_feature(self):
		ref = pd.read_csv('../tests/data/single_ts-feature.csv')
		inp = pd.read_csv('../tests/data/extracted_features.csv')
		case = ts_feature(inp, "autocorrelation")
		self.assertEqual((ref.all() == case.all()).all(), True, "Results don't match")

	def test_all_ts_features(self):
		ref = pd.read_csv('../tests/data/all_ts_features.csv')
		inp = pd.read_csv('../tests/data/extracted_features.csv')
		case = ts_all_features(inp)
		self.assertEqual((ref.all() == case.all()).all(), True, "Results don't match")

	def test_medoid_calculation(self):
		# Read in CSV file-
		data_d = {
			'time': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
			'animal_id': [312, 511, 607, 811, 905, 312, 511, 607, 811, 905],
			'x': [405.29, 369.99, 390.33, 445.15, 366.06, 405.31, 370.01,
				  390.25, 445.48, 365.86],
			'y': [417.76, 428.78, 405.89, 411.94, 451.76, 417.37, 428.82,
				  405.89, 412.26, 451.76],
			'distance_to_centroid': [11.331, 25.975, 18.052, 51.049, 40.901,
									 11.523, 25.983, 18.074, 51.283, 41.062],
			'medoid': [312, 312, 312, 312, 312, 312, 312, 312, 312, 312]
		}

		data = pd.DataFrame(data_d)

		# Compute medoid using function in 'medoid_calculation.py' Python file-
		data_medoid = medoid_computation(data)

		df_calculated_data = data_medoid.iloc[:10, :]

		computed_data = {
			'time': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
			'animal_id': [312, 511, 607, 811, 905, 312, 511, 607, 811, 905],
			'x': [405.29, 369.99, 390.33, 445.15, 366.06, 405.31, 370.01,
				  390.25, 445.48, 365.86],
			'y': [417.76, 428.78, 405.89, 411.94, 451.76, 417.37, 428.82,
				  405.89, 412.26, 451.76],
			'distance_to_centroid': [11.331, 25.975, 18.052, 51.049, 40.901,
									 11.523, 25.983, 18.074, 51.283, 41.062],
			'medoid': [312, 312, 312, 312, 312, 312, 312, 312, 312, 312]
		}

		df_computed_data = pd.DataFrame(computed_data)

		assert_frame_equal(df_calculated_data, df_computed_data)
	# self.assertEqual(df_calculated_data, df_computed_data)



if __name__ == '__main__':
    unittest.main()
