import os
import unittest
import pandas as pd
import numpy as np
import math
import pickle
from pandas.testing import assert_frame_equal

os.chdir("C:/Users/lukas/Dropbox/Movekit/")
from src.movekit.feature_extraction import *
os.chdir("C:/Users/lukas/Dropbox/Movekit/tests")


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
		with open('../tests/data/dict_groups.pickle', 'rb') as handle:
			ref = pickle.load(handle)
		case = grouping_data(inp)
		for i in ref.keys():
			self.assertTrue((ref[i].all() == case[i].all()).all(), "Results don't match")


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
		for i in ref.keys():
			self.assertTrue((ref[i].all() == case[i].all()).all(), "Results don't match")

	def test_compute_average_speed(self):
		with open('../tests/data/av_speed.pickle', 'rb') as handle:
			ref = pickle.load(handle)
		with open('../tests/data/dist_and_dir.pickle', 'rb') as handle:
			inp = pickle.load(handle)
		case = compute_average_speed(inp, fps = 10)
		for i in ref.keys():
			self.assertTrue((ref[i].all() == case[i].all()).all(), "Results don't match")

	def test_average_acceleration(self):
		with open('../tests/data/av_accel.pickle', 'rb') as handle:
			ref = pickle.load(handle)
		with open('../tests/data/av_speed.pickle', 'rb') as handle:
			inp = pickle.load(handle)
		case = compute_average_acceleration(inp, fps = 10)
		for i in ref.keys():
			self.assertTrue((ref[i].all() == case[i].all()).all(), "Results don't match")

	def test_computing_stops(self):
		with open('../tests/data/stops.pickle', 'rb') as handle:
			ref = pickle.load(handle)
		with open('../tests/data/av_accel.pickle', 'rb') as handle:
			inp = pickle.load(handle)
		case = computing_stops(inp, threshold_speed = 0.5)
		for i in ref.keys():
			self.assertTrue((ref[i].all() == case[i].all()).all(), "Results don't match")

	def test_medoid_computation(self):
		inp = pd.read_csv("../tests/data/records.csv")
		ref = pd.read_csv("../tests/data/medoids.csv")
		case = medoid_computation(inp)
		self.assertTrue((case.all()==ref.all()).all, "Results don't match")

	def test_compute_absolute_features(self):
		with open('../tests/data/stops.pickle', 'rb') as handle:
			ref = pickle.load(handle)
		with open('../tests/data/dict_groups.pickle', 'rb') as handle:
			inp = pickle.load(handle)
		case = compute_absolute_features(inp, fps = 10 , stop_threshold = 0.5)
		for i in ref.keys():
			self.assertTrue((ref[i].all() == case[i].all()).all(), "Results don't match")

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

if __name__ == '__main__':
    unittest.main()
