

import pandas as pd
import numpy as np
# from pandas.api.types import is_numeric_dtype, is_string_dtype
# from pandas.io.common import EmptyDataError
import math


def grouping_data(processed_data):
    '''
    A function to group all values for each 'animal_id' attribute

    Input is 'processed_data' which is processed Pandas DataFrame
    Returns a dictionary where- key is animal_id & value is Pandas DataFrame
    for that 'animal_id'
    '''
    # A dictionary object to hold all groups obtained using group by-
    data_animal_id_groups = {}

    # Group by using 'animal_id' attribute-
    data_animal_id = processed_data.groupby('animal_id')

    # Get each animal_id's data from grouping performed-
    for animal_id in data_animal_id.groups.keys():
        data_animal_id_groups[animal_id] = data_animal_id.get_group(animal_id)

    # To reset index for each group-
    for animal_id in data_animal_id_groups.keys():
        data_animal_id_groups[animal_id].reset_index(drop=True, inplace=True)

    # Add additional attributes/columns to each groups-
    for aid in data_animal_id_groups.keys():
        data = [0 for x in range(data_animal_id_groups[aid].shape[0])]

        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            distance=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            average_speed=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            average_acceleration=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            direction=data)

    return data_animal_id_groups


def compute_distance_and_direction(data_animal_id_groups):
    '''
    Calculate metric distance and direction-

    Calculate the metric distance between two consecutive time frames/time stamps
    for each moving entity (in this case, fish)
    '''
    # start_time = time.time()

    for aid in data_animal_id_groups.keys():
        print("\nComputing Distance & Direction for Animal ID = {0}\n".format(
            aid))

        # for i in range(1, animal_id.shape[0] - 1):
        for i in range(1, data_animal_id_groups[aid].shape[0] - 1):
            # print("Current i = ", i)

            x1 = data_animal_id_groups[aid].iloc[i, 2]
            y1 = data_animal_id_groups[aid].iloc[i, 3]
            x2 = data_animal_id_groups[aid].iloc[i + 1, 2]
            y2 = data_animal_id_groups[aid].iloc[i + 1, 3]

            # Compute distance between 2 points-
            distance = math.sqrt(
                math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))

            # Compute the direction in DEGREES-
            direction = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
            if math.isnan(direction):
                data_animal_id_groups[aid].loc[i, 'direction'] = 0
                # animal_id.loc[i, 'Direction'] = 0
            else:
                data_animal_id_groups[aid].loc[i, 'direction'] = direction
                # animal_id.loc[i, 'Direction'] = direction

            # Insert computed distance to column/attribute 'Distance'-
            # animal_id.loc[i, 'Distance'] = distance
            data_animal_id_groups[aid].loc[i, 'distance'] = distance

    # end_time = time.time()
    # print("\nTime taken to create distance & direction data = {0:.4f} seconds\n\n".format(
    # end_time - start_time))
    # Time taken to create distance & direction data = 1013.1692 seconds

    return data_animal_id_groups


def input_heading_angle(path_to_file):
    '''
    Function to accept CSV file having attribute 'heading_angle' which
    contains pre-computed direction for each animal

    Input: 'data' which is a CSV file
    Return: processed CSV file as Pandas DataFrame
    '''

    try:

        if path_to_file[-3:] == 'csv':
            data = pd.read_csv(path_to_file)
        else:
            data = pd.read_csv(path_to_file + '.csv')

        # change column names all to lower case values
        data.columns = map(str.lower, data.columns)

        # check if all required columns are there in the right format
        if 'time' in data and 'animal_id' in data and 'x' in data and 'y' in data:
            # Check if 'time' attribute is integer-
            # if is_numeric_dtype(data['time']):
            if np.issubdtype(data['time'].dtype, np.number):
                data.sort_values('time', ascending=True, inplace=True)

            # Check if 'time' attribute is string-
            # elif is_string_dtype(data['time']):
            elif np.issubdtype(data['time'].dtype, np.number) == False:
                data['time'] = pd.to_datetime(data['time'])
                data.sort_values('time', ascending=True, inplace=True)

            # Check if 'heading_angle' attribute is given in CSV file-
            if 'heading_angle' in data and np.issubdtype(data['heading_angle'].dtype, np.number):
                print("\n'heading_angle' attribute is found (numeric type) and will be processed\n")
                # do nothing, as 'heading_angle' attribute exists
            else:
                print("\nWARNING: 'heading_angle' attribute is not found in the given CSV data file. Continuing without it!\n")

            return data

    except FileNotFoundError:
        print("Your file below could not be found.\nPath given: {0}\n\n".format(path_to_file))
    """
    except EmptyDataError:
        print('Your file is empty, has no header, or misses some required columns.')
    """


file_path = input("\nEnter complete path to CSV file: ")

