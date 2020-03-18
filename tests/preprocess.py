# Missing:

# linear_interpolation
# plot_missing_values
# preprocess
# print_missing
# print_duplicate
# grouping_data
# filter_dataframe
# replace_parts_animal_movement
# resample_systematic
# resample_random
# split_trajectories
# split_trajectories_fuzzy segmentation
import unittest
import pandas as pd
import numpy as np
import math
import pickle


# from pandas.util.testing import assert_frame_equal
from pandas.testing import assert_frame_equal
import sys
#os.chdir("C:/Users/lukas/Dropbox/Movekit/")

from pandas.errors import EmptyDataError
from src.movekit.preprocess import *
idlist = [312, 511, 607, 811, 905]


class TestPreprocess(unittest.TestCase):

    def test_filter_dataframe(self):
        inp = pd.read_csv('../tests/data/records.csv')
        ref = pd.read_csv('../tests/data/filtered.csv')
        case = filter_dataframe(inp, 1,3)
        self.assertTrue((ref.all()==inp.all()).all(), "Results don't match")

    def test_replace_parts_movement(self):
        with open('../tests/data/dict_groups.pickle', 'rb') as handle:
            inp = pickle.load(handle)
        with open('../tests/data/replaced.pickle', 'rb') as handle:
            ref = pickle.load(handle)
            arr_index = np.array([10, 20, 200])
        case = replace_parts_animal_movement(inp, 811, arr_index, replacement_value_x = 100, replacement_value_y = 90)
        self.assertTrue((ref[811].all() == case[811].all()).all(), "Results don't match")

    def test_resample_systematic(self):
        with open('../tests/data/dict_groups.pickle', 'rb') as handle:
            inp = pickle.load(handle)
        with open('../tests/data/sys_sample.pickle', 'rb') as handle:
            ref = pickle.load(handle)
        case = resample_systematic(inp, 10)
        for i in ref.keys():
            self.assertTrue((ref[i].all() == case[i].all()).all(), "Results don't match")


    def test_resample_random(self):
        with open('../tests/data/dict_groups.pickle', 'rb') as handle:
            inp = pickle.load(handle)
        case = resample_random(inp, 10)
        for i in case.keys():
            self.assertEqual(len(case[i]), 10, "Results don't match")

    def test_split_trajectories(self):
        with open('../tests/data/dict_groups.pickle', 'rb') as handle:
            inp = pickle.load(handle)
        with open('../tests/data/split_traj.pickle', 'rb') as handle:
            ref = pickle.load(handle)
        case = split_trajectories_fuzzy_segmentation(inp, 3)
        for i in ref.keys():
            self.assertTrue((case[i].all() == ref[i].all()).all(), "Results don't match")


if __name__ == '__main__':
    unittest.main()
