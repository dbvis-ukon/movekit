

import unittest
import math
import pandas as pd
import numpy as np


class Test_Data(unittest.TestCase):

    def test_average_speed(self):
        fps = 10
        # result = pd.read_csv("Complete-Processed_Data-fps_10-02_May_2019.csv")
        result = pd.read_csv("Completely_Processed_Data-fps_10_Part-2.csv")

        # animal_id = 312
        # row number = 3
        index = 3
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 0.15077, "Shoud be 0.15077")

        index = 2541
        i = index - 2

        # print("\nindex = {0} and i = {1}\n".format(index, i))
        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 1.7527, "Shoud be 1.7527")

        index = 35042
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 1.2837, "Shoud be 1.2837")

        index = 43165
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 1.95778, "Shoud be 1.95778")

        index = 43193
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 5.09769, "Shoud be 5.09769")

        # Animal ID = 511
        index = 44917
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 1.70317, "Shoud be 1.70317")

        index = 63784
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 3.94876, "Shoud be 3.94876")

        index = 83097
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 1.82929, "Shoud be 1.82929")

        index = 86394
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 2.24186, "Shoud be 2.24186")

        # Animal ID = 607
        index = 86457
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 0.15985, "Shoud be 0.15985")

        index = 114532
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 4.6443, "Shoud be 4.6443")

        index = 115018
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 3.24535, "Shoud be 3.24535")

        index = 127553
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 3.21445, "Shoud be 3.21445")

        index = 129595
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 2.48661, "Shoud be 2.48661")

        # Animal ID = 811

        index = 129606
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 0.51454, "Shoud be 0.51454")

        index = 156747
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 4.9209, "Shoud be 4.9209")

        index = 169768
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 2.78231, "Shoud be 2.78231")

        index = 172796
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 3.75974, "Shoud be 3.75974")

        # Animal ID = 905

        index = 172916
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 0.12779, "Shoud be 0.12779")

        index = 189081
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 1.72438, "Shoud be 1.72438")

        index = 206144
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 5.79925, "Shoud be 5.79925")

        index = 206823
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 2.46544, "Shoud be 2.46544")

        index = 215997
        i = index - 2

        avg_speed = round(
            sum(result.loc[i:(i + fps - 1), "Distance"]) / fps, 5)
        self.assertEqual(avg_speed, 3.65608, "Shoud be 3.65608")

    def test_average_acceleration(self):
        fps = 10
        # result = pd.read_csv("Complete-Processed_Data-fps_10-02_May_2019.csv")
        result = pd.read_csv("Completely_Processed_Data-fps_10_Part-2.csv")

        # Animal ID = 312
        # row number = 3
        index = 3
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        # print("\nFor index = {0}, avg_acceleration = {1:.5f}\n".format(index, avg_acceleration))

        self.assertEqual(avg_acceleration, -0.01727, "Should be -0.01727")

        index = 1234
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration, -0.00725, "Should be -0.00725")

        index = 26063
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration, 0.0982, "Should be 0.0982")

        index = 43193
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration, -5.09769, "Should be -5.09769")

        # Animal ID = 511

        index = 43204
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration, 0.00622, "Should be 0.00622")

        index = 49867
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration, -0.05845, "Should be -0.05845")

        index = 63338
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  -0.01581, "Should be  -0.01581")

        index = 86393
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  -0.2445, "Should be  -0.2445")

        index = 86394
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  -2.24186, "Should be  -2.24186")

        # Animal ID = 607

        index = 86406
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  -0.005, "Should be  -0.005")

        index = 88094
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  0.07114, "Should be  0.07144")

        index = 106887
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  -0.06882, "Should be  -0.06882")

        index = 123964
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  0.05577, "Should be  0.05577")

        index = 123964
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  0.05577, "Should be  0.05577")

        index = 127558
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  0.1769, "Should be  0.1769")

        # Animal ID = 811

        index = 127558
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  0.1769, "Should be  0.1769")

        index = 144159
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  -0.06426, "Should be  -0.06426")

        index = 151357
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  0.03585, "Should be  0.03585")

        index = 167519
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  0.17758, "Should be  0.17758")

        index = 172781
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  0.33583, "Should be  0.33583")

        # Animal ID = 905

        index = 172819
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  -0.00106, "Should be  -0.00106")

        index = 188276
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  -0.09047, "Should be  -0.09047")

        index = 199828
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  -0.27189, "Should be  -0.27189")

        index = 211538
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  0.08766, "Should be  0.08766")

        index = 215996
        i = index - 2

        avg_acceleration = round(
            (result.loc[i + 1, "Average_Speed"] - result.loc[i, "Average_Speed"]), 5)
        self.assertEqual(avg_acceleration,  -0.36341, "Should be -0.36341")

    def test_direction(self):

        result = pd.read_csv("Completely_Processed_Data-fps_10_Part-2.csv")

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

        self.assertEqual(direction,  -92.7263, "Should be -92.7263")

        index = 28751
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

        self.assertEqual(direction,  -158.7723, "Should be -158.7723")

        index = 31360
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

        self.assertEqual(direction,  -162.3818, "Should be -162.3818")

        # Animal ID = 511

        index = 52106
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

        self.assertEqual(direction,  -141.4908, "Should be -141.4908")

        index = 69619
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

        self.assertEqual(direction,  7.6961, "Should be 7.6961")

        index = 84505
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

        self.assertEqual(direction,  35.8377, "Should be 35.8377")

        # Animal ID = 607

        index = 86627
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

        self.assertEqual(direction,  176.5318, "Should be 176.5318")

        index = 103299
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

        self.assertEqual(direction,  156.1048, "Should be 156.1048")

        index = 124863
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

        self.assertEqual(direction,  157.3801, "Should be 157.3801")

        # Animal ID = 811

        index = 132043
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

        self.assertEqual(direction,  40.9144, "Should be 40.9144")

        index = 149561
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

        self.assertEqual(direction,  171.5626, "Should be 171.5626")

        index = 168876
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

        self.assertEqual(direction,  -71.2135, "Should be -71.2135")

        # Animal ID = 905

        index = 183235
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

        self.assertEqual(direction,  -37.9477, "Should be -37.9477")

        index = 203893
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

        self.assertEqual(direction,  22.8337, "Should be 22.8337")

        index = 215650
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

        self.assertEqual(direction,  -18.0490, "Should be -18.0490")


if __name__ == '__main__':
    unittest.main()
