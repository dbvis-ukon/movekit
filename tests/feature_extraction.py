

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
from src.movekit.feature_extraction import medoid_computation, grouping_data, compute_average_speed, euclidean_dist
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
		with open('data/dict_groups.pkl', 'rb') as handle:
			ref = pickle.load(handle)
			case = grouping_data(inp)
		self.assertEqual((ref[312].all() == case[312].all()).all(), True, "Results don't match")
		self.assertEqual((ref[511].all() == case[511].all()).all(), True, "Results don't match")
		self.assertEqual((ref[607].all() == case[607].all()).all(), True, "Results don't match")
		self.assertEqual((ref[811].all() == case[811].all()).all(), True, "Results don't match")
		self.assertEqual((ref[905].all() == case[905].all()).all(), True, "Results don't match")

	def test_grouping_data(self):
		"""
		Testing grouping data function iteratively. Not work yet.
		:return:
		"""
		inp = pd.read_csv("../tests/data/records.csv")
		with open('data/dict_groups.pkl', 'rb') as handle:
			ref = pickle.load(handle)
			case = grouping_data(inp)
		for aid in inp.keys():
			self.assertEqual((ref[aid].all() == case[aid].all()).all(), True, "Results don't match")


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
		case = case.rename(columns = str)
		self.assertEqual((ref.reset_index(drop=True).all() == case.reset_index(drop=True).all()).all(), True,
						 "Results don't match")




	def test_average_speed(self):
		#fps = 10
		inp = pd.read_csv("../tests/data/Completely_Processed_Data-fps_10.csv")
		result = grouping_data(inp)

		# Animal_id = 312

		# row number = 13
		index = 13
		i = index - 9

		avg_speed = compute_average_speed(result, fps = 10)
		self.assertEqual(avg_speed, round(result[312]["avg_speed"][13]), "Results don't match")

		index = 160
		i = index - 2

		# print("\nindex = {0} and i = {1}\n".format(index, i))
		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 1.58274, "Shoud be 1.58274")



		index = 500
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 1.06081, "Shoud be 1.06081")


		index = 700
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 1.72602, "Shoud be 1.72602")


		index = 1100
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 5.57793, "Shoud be 5.57793")


		# Animal ID = 511

		index = 44917
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 1.71695, "Shoud be 1.71695")


		index = 63784
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 3.72608, "Shoud be 3.72608")


		index = 83097
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 1.98837, "Shoud be 1.98837")


		index = 86394
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 2.48636, "Shoud be 2.48636")


		# Animal ID = 607
		index = 86457
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 0.13292, "Shoud be 0.13292")


		index = 114532
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 4.79641, "Shoud be 4.79641")


		index = 115018
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 3.13758, "Shoud be 3.13758")


		index = 127553
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 3.2756, "Shoud be 3.2756")


		index = 129595
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 2.69573, "Shoud be 2.69573")



		# Animal ID = 811

		index = 129606
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 0.50008, "Shoud be 0.50008")


		index = 156747
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 4.87471, "Shoud be 4.87471")


		index = 169768
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 2.98046, "Shoud be 2.98046")


		index = 172796
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 4.153, "Shoud be 4.153")



		# Animal ID = 905

		index = 172916
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 0.12116, "Shoud be 0.12116")


		index = 189081
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 1.93891, "Shoud be 1.93891")


		index = 206144
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 5.6621, "Shoud be 5.6621")


		index = 206823
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 2.37117, "Shoud be 2.37117")


		index = 215997
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 4.01949, "Shoud be 4.01949")




	def test_average_acceleration(self):
		fps = 10
		result = pd.read_csv("../tests/data/Completely_Processed_Data-fps_10_Part-2.csv")


		# Animal ID = 312 'the label [1232] is not in the [index]'

		index = 1234
		i = index - 2

		'''
		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i + 1, 'average_speed']), 5)
		'''
		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, 0.0033, "Should be 0.0033")


		index = 26063
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, -0.01414, "Should be -0.01414")


		index = 43193
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, 0.0174, "Should be 0.0174")



		# Animal ID = 511

		index = 49867
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, -0.01863, "Should be -0.01863")


		index = 63338
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, -0.00684, "Should be  -0.00684")


		index = 86393
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, 0.01868, "Should be  0.01868")


		index = 86394
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, 0.01984, "Should be  0.01984")



		# Animal ID = 607

		index = 88094
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, 0.00403, "Should be  0.00403")


		index = 106887
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, 0.01233, "Should be  0.01233")


		index = 123964
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, -0.00467, "Should be  -0.00467")


		index = 127558
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, 0.01082, "Should be 0.01082")



		# Animal ID = 811

		index = 144159
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, -0.00891, "Should be -0.00891")


		index = 151357
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, 0.03168, "Should be 0.03168")


		index = 167519
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, -0.02405, "Should be -0.02405")


		index = 172781
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, -0.00111, "Should be -0.00111")



		# Animal ID = 905

		index = 172819
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, -0.0007, "Should be  -0.0007")


		index = 188276
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, 0.01052, "Should be  0.01052")


		index = 199828
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, 0.02498, "Should be  0.02498")


		index = 211538
		i = index - 2

		avg_acceleration = round((result.loc[i, 'average_speed'] -
			result.loc[i - 1, 'average_speed']), 5)
		avg_acceleration = round((avg_acceleration / fps), 5)
		self.assertEqual(avg_acceleration, 0.01598, "Should be  0.01598")


	def test_direction(self):

		result = pd.read_csv("../tests/data/Completely_Processed_Data-fps_10_Part-2.csv")

		# Animal ID = 312

		index = 4
		i = index - 2

		x1 = result.iloc[i, 2]
		y1 = result.iloc[i, 3]
		x2 = result.iloc[i + 1, 2]
		y2 = result.iloc[i + 1, 3]

		# Compute the direction in DEGREES-
		# direction = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
		direction = round(math.degrees(math.atan2((y2 - y1), (x2 - x1))), 4)

		if (x2 - x1) == 0:
			direction = 0
		elif math.isnan(direction):
			direction = 0

		# self.assertEqual(direction, -92.7263, "Should be -92.7263")
		self.assertEqual(direction, round(result.loc[i + 1, 'direction'], 4), "Should be -92.7263")




		index = 100
		i = index - 2

		x1 = result.iloc[i, 2]
		y1 = result.iloc[i, 3]
		x2 = result.iloc[i + 1, 2]
		y2 = result.iloc[i + 1, 3]


		# Compute the direction in DEGREES-
		# direction = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
		direction = round(math.degrees(math.atan2((y2 - y1), (x2 - x1))), 4)

		if (x2 - x1) == 0:
			direction = 0
		elif math.isnan(direction):
			direction = 0

		# self.assertEqual(direction, -92.7263, "Should be -92.7263")
		self.assertEqual(direction, round(result.loc[i + 1, 'direction'], 4), "Should be -53.7462")




		index = 70
		i = index - 2

		x1 = result.iloc[i, 2]
		y1 = result.iloc[i, 3]
		x2 = result.iloc[i + 1, 2]
		y2 = result.iloc[i + 1, 3]


		# Compute the direction in DEGREES-
		# direction = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
		direction = round(math.degrees(math.atan2((y2 - y1), (x2 - x1))), 4)

		if (x2 - x1) == 0:
			direction = 0
		elif math.isnan(direction):
			direction = 0

		# self.assertEqual(direction, -92.7263, "Should be -92.7263")
		self.assertEqual(direction, round(result.loc[i + 1, 'direction'], 4), "Should be -45.0")




		index = 80
		i = index - 2

		x1 = result.iloc[i, 2]
		y1 = result.iloc[i, 3]
		x2 = result.iloc[i + 1, 2]
		y2 = result.iloc[i + 1, 3]


		# Compute the direction in DEGREES-
		# direction = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
		direction = round(math.degrees(math.atan2((y2 - y1), (x2 - x1))), 4)

		if (x2 - x1) == 0:
			direction = 0
		elif math.isnan(direction):
			direction = 0

		# self.assertEqual(direction, -92.7263, "Should be -92.7263")
		self.assertEqual(direction, round(result.loc[i + 1, 'direction'], 4), "Should be -63.4349")

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
