

import unittest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
# from medoid_calculation import medoid_computation
from movekit.feature_extraction.medoid_calculation import medoid_computation
# import time


class Test_Medoid_Calculation(unittest.TestCase):
    """
    Unit testing class for computation of medoid feature for fish dataset
    """
    def test_case1(self):
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

    # start = time.time()
    unittest.main()
    # end = time.time()
    # print("\nTime taken for medoid calculation unit testing = {0:.4f} ms\n".format((end - start) * 1000))


