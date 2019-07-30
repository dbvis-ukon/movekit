"""
  Extract features for individual moving entities
  Author: Arjun Majumdar, Eren Cakmak
  Created: July, 2019
"""
import math
# import time

import pandas as pd

from tsfresh import select_features
from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute


def compute_absolute_features(data_animal_id_groups):
    '''
   Calculate absolute features for the data animal group-
   '''
    direction_distance_data = compute_distance_and_direction(
        data_animal_id_groups)
    avg_speed_data = compute_average_speed(direction_distance_data, 3)
    avg_acceleration_data = compute_average_acceleration(avg_speed_data, 3)
    print(avg_acceleration_data)
    return avg_acceleration_data


def compute_distance_and_direction(data_animal_id_groups):
    '''
    Calculate metric distance and direction-

    Calculate the metric distance between two consecutive time frames/time stamps
    for each moving entity (in this case, fish)
    '''
    # start_time = time.time()

    for aid in data_animal_id_groups.keys():
        print(
            "\nComputing Distance & Direction for Animal ID = {0}\n".format(aid))

        # for i in range(1, animal_id.shape[0] - 1):
        for i in range(1, data_animal_id_groups[aid].shape[0] - 1):
            # print("Current i = ", i)

            x1 = data_animal_id_groups[aid].iloc[i, 2]
            y1 = data_animal_id_groups[aid].iloc[i, 3]
            x2 = data_animal_id_groups[aid].iloc[i + 1, 2]
            y2 = data_animal_id_groups[aid].iloc[i + 1, 3]

            # Compute distance between 2 points-
            distance = math.sqrt(math.pow((x2 - x1), 2) +
                                 math.pow((y2 - y1), 2))

            # Compute the direction in DEGREES-
            direction = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
            if math.isnan(direction):
                data_animal_id_groups[aid].loc[i, 'Direction'] = 0
                # animal_id.loc[i, 'Direction'] = 0
            else:
                data_animal_id_groups[aid].loc[i, 'Direction'] = direction
                # animal_id.loc[i, 'Direction'] = direction

            # Insert computed distance to column/attribute 'Distance'-
            # animal_id.loc[i, 'Distance'] = distance
            data_animal_id_groups[aid].loc[i, 'Distance'] = distance

    # end_time = time.time()
    # print("\nTime taken to create distance & direction data = {0:.4f} seconds\n\n".format(
        # end_time - start_time))
    # Time taken to create distance & direction data = 1013.1692 seconds

    return data_animal_id_groups


def compute_average_speed(data_animal_id_groups, fps):
    '''
    Average Speed-

    A function to compute average speed of an animal based on fps
    (frames per second) parameter. Calculate the average speed of a mover,
    based on the pandas dataframe and a frames per second (fps) parameter

    Formula used-
    Average Speed = Total Distance Travelled / Total Time taken
    '''

    # start_time = time.time()

    for aid in data_animal_id_groups.keys():
        print("\nComputing Average Speed for Animal ID = {0}\n".format(aid))

        # for i in range (1, animal_id.shape[0] - fps + 1):
        for i in range(1, data_animal_id_groups[aid].shape[0] - fps + 1):
            # print("Current i = ", i)
            # Current i =  45044
            # KeyError: 45046

            tot_dist = 0  # total distance travelled

            for j in range(i, i + fps):
                # tot_dist += animal_id.loc[j, "Distance"]
                tot_dist += data_animal_id_groups[aid].loc[j, "Distance"]

            # animal_id.loc[i, "Average_Speed"] = (tot_dist / fps)
            data_animal_id_groups[aid].loc[i,
                                           "Average_Speed"] = (tot_dist / fps)

    # end_time = time.time()
    # print("\nTime taken to create Average Speed data = {0:.4f} seconds.\n".format(
    #     end_time - start_time))

    return data_animal_id_groups


def compute_average_acceleration(data_animal_id_groups, fps):
    '''
    A function to compute average acceleration of an animal based on fps
    (frames per second) parameter.

    Formulas used are-
    Average Acceleration = (Final Speed - Initial Speed) / Total Time Taken
    '''

    # start_time = time.time()

    for aid in data_animal_id_groups.keys():
        print("\nComputing Average Speed for Animal ID = {0}\n".format(aid))

        # for i in range (1, animal_id.shape[0] - fps + 1):
        for i in range(1, data_animal_id_groups[aid].shape[0] - fps + 1):
            # print("Current i = ", i)

            avg_speed = 0

            # Calculating Average Speed-
            avg_speed = data_animal_id_groups[aid].loc[i, 'Average_Speed'] - \
                data_animal_id_groups[aid].loc[i + 1, 'Average_Speed']
            # avg_speed = animal_id.loc[i, "Average_Speed"] - animal_id.loc[i + 1, "Average_Speed"]
            # print("\navg_speed = {0:.4f}\n".format(avg_speed))
            # animal_id.loc[i, "Average_Acceleration"] = (avg_speed / fps)
            data_animal_id_groups[aid].loc[i,
                                           'Average_Acceleration'] = (avg_speed / fps)

    # end_time = time.time()
    # print("\nTime taken to create Average Acceleration data = {0:.4f} seconds.\n".format(
    #     end_time - start_time))
    # Total time taken = 37.8197 seconds.

    # Concatenate all Pandas DataFrame into one-
    result = pd.concat(data_animal_id_groups[aid]
                       for aid in data_animal_id_groups.keys())

    # Reset index-
    result.reset_index(drop=True, inplace=True)

    return result


def time_series_analyis(data):
    extracted_features = extract_features(
        data, column_id='animal_id', column_sort='time')
    impute(extracted_features)
    return extracted_features
