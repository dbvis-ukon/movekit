

import unittest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
from real_positive_acceleration import real_positive_acceleration
# from absolute import grouping_data


def group_data(data_frame):
    data_animal_id_groups = {}

    data_animal_id = data_frame.groupby('animal_id')

    for aid in data_animal_id.groups.keys():
        data_animal_id_groups[aid] = data_animal_id.get_group(aid)

    for aid in data_animal_id_groups.keys():
        data_animal_id_groups[aid].reset_index(drop = True, inplace = True)

    return data_animal_id_groups


class Test_Real_Positive_Acceleration(unittest.TestCase):
    def test_case1(self):

        calculated_data = {
'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'animal_id': [312, 312, 312, 312, 312, 312, 312, 312, 312, 312],
 'x': [405.29,
  405.31,
  405.31,
  405.3,
  405.29,
  405.27,
  405.27,
  405.27,
  405.31,
  405.38],
 'y': [417.76,
  417.37,
  417.07,
  416.86,
  416.71,
  416.61,
  416.54,
  416.49,
  416.37,
  416.27],
 'distance': [0.0,
  0.30000000000001137,
  0.2102379604162655,
  0.1503329637837625,
  0.10198039027182984,
  0.06999999999999318,
  0.05000000000001137,
  0.12649110640674596,
  0.12206555615735172,
  0.16124515496595124],
 'average_speed': [0.0,
  0.16651026289437248,
  0.11651026289437247,
  0.09976089209246856,
  0.0941074105671864,
  0.10596036350601072,
  0.13504168196308902,
  0.1504975260858034,
  0.15119930480444546,
  0.17105808081533888],
 'average_acceleration': [0.0,
  0.010000000000000005,
  0.003349874160380781,
  0.0011306963050564307,
  -0.002370590587764859,
  -0.005816263691415663,
  -0.003091168824542878,
  -0.00014035574372840974,
  -0.003971755202178685,
  -0.0008932177016346797],
 'direction': [0.0,
  -90.0,
  87.27368900609596,
  86.18592516571398,
  78.69006752595475,
  -90.0,
  -90.0,
  -71.56505117706985,
  -55.007979801450084,
  -60.25511870306028],
 'real_positive_acceleration': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0]
            }

        # df_calculated_data = pd.DataFrame(calculated_data)
        data = pd.DataFrame(calculated_data)

        # Group data-
        data_grouped = group_data(data)

        data_real_pos_acc = real_positive_acceleration(data_grouped)


        computed_data = {
'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'animal_id': [312, 312, 312, 312, 312, 312, 312, 312, 312, 312],
 'x': [405.29,
  405.31,
  405.31,
  405.3,
  405.29,
  405.27,
  405.27,
  405.27,
  405.31,
  405.38],
 'y': [417.76,
  417.37,
  417.07,
  416.86,
  416.71,
  416.61,
  416.54,
  416.49,
  416.37,
  416.27],
 'distance': [0.0,
  0.30000000000001137,
  0.2102379604162655,
  0.1503329637837625,
  0.10198039027182984,
  0.06999999999999318,
  0.05000000000001137,
  0.12649110640674596,
  0.12206555615735172,
  0.16124515496595124],
 'average_speed': [0.0,
  0.16651026289437248,
  0.11651026289437247,
  0.09976089209246856,
  0.0941074105671864,
  0.10596036350601072,
  0.13504168196308902,
  0.1504975260858034,
  0.15119930480444546,
  0.17105808081533888],
 'average_acceleration': [0.0,
  0.010000000000000005,
  0.003349874160380781,
  0.0011306963050564307,
  -0.002370590587764859,
  -0.005816263691415663,
  -0.003091168824542878,
  -0.00014035574372840974,
  -0.003971755202178685,
  -0.0008932177016346797],
 'direction': [0.0,
  -90.0,
  87.27368900609596,
  86.18592516571398,
  78.69006752595475,
  -90.0,
  -90.0,
  -71.56505117706985,
  -55.007979801450084,
  -60.25511870306028],
 'real_positive_acceleration': [0.0,
  0.010000000000000005,
  0.003349874160380781,
  0.0011306963050564307,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0]
             }


        df_computed_data = pd.DataFrame(computed_data)

        # assert_frame_equal(df_calculated_data, df_computed_data)
        assert_frame_equal(data_real_pos_acc, df_computed_data,
                "Pandas DataFrames should be equal!")
        # self.assertEqual(df_calculated_data, df_computed_data)


if __name__ == '__main__':
	unittest.main()


# To execute the testing-
# $ python test_real_positive_acceleration.py -v


