

import os
import unittest
import pandas as pd
import numpy as np
import math

# from pandas.util.testing import assert_frame_equal
from pandas.testing import assert_frame_equal

# from pandas.io.common import EmptyDataError
from pandas.errors import EmptyDataError


os.chdir("../src/")


import movekit as mkit
import movekit


from movekit.feature_extraction_combined import grouping_data
from movekit.feature_extraction_combined import compute_distance_and_direction
from movekit.feature_extraction_combined import compute_average_speed
from movekit.feature_extraction_combined import compute_average_acceleration

# from movekit.io_combined import read_data
# from movekit.io_combined import parse_csv
# from movekit.io_combined import parse_excel
# from movekit.preprocessing_combined import preprocessing_methods
# from movekit.feature_extraction_combined import feature_extraction_methods
# from movekit.plot import plotting_methods


class Test_Feature_Extraction(unittest.TestCase):
	'''
	Unit Tests for Feature Extraction
	'''

	def test_average_speed(self):
		fps = 10
		result = pd.read_csv("../tests/data/Completely_Processed_Data-fps_10_Part-2.csv")


		# Animal_id = 312

		# row number = 3
		index = 3
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 0.16829, "Shoud be 0.16829")
		

		index = 2541
		i = index - 2

		# print("\nindex = {0} and i = {1}\n".format(index, i))
		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 1.58274, "Shoud be 1.58274")



		index = 35042
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 1.06081, "Shoud be 1.06081")


		index = 43165
		i = index - 2

		avg_speed = round(
			sum(result.loc[i:(i + fps - 1), "distance"]) / fps, 5)
		self.assertEqual(avg_speed, 1.72602, "Shoud be 1.72602")


		index = 43193
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


		# Animal ID = 312

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





if __name__ == '__main__':
    unittest.main()
