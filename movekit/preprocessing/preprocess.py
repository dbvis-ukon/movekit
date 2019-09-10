"""
  Preprocess the data frame
  Author: Arjun Majumdar, Eren Cakmak
  Created: July, 2019
"""

import pandas as pd
import numpy as np


def clean(data):
    '''
    A function to perform data preprocessing
    Expects 'data' as input which is the Pandas DataFrame to be processed
    '''
    # Print the number of missing values per column
    print_missing(data)

    # Drop columns with  missing values for 'time'  and 'animal_id'
    data.dropna(subset=['animal_id', 'time'], inplace=True)
    # Change column type of animal_id and time
    data['animal_id'] = data['animal_id'].astype(np.int64)
    data['time'] = data['time'].astype(np.int64)

    # Print duplicate rows
    print_duplicate(data)
    # Remove the duplicated rows found above
    data.drop_duplicates(subset=['animal_id', 'time'], inplace=True)

    return data


def print_missing(df):
    '''
    Print the missing values for each column
    '''
    print('Missing values:\n', df.isnull().sum().sort_values(ascending=False))


def print_duplicate(df):
    '''
    Print the duplicate rows 
    '''
    dup = df[df.duplicated(['time', 'animal_id'])]
    print("Removed duplicate rows based on the columns 'animal_id' and 'time' column are:", dup, sep='\n')


def grouping_data(processed_data):
    '''
    A function to group all values for each 'animal_id'
    Input is 'processed_data' which is processed Pandas DataFrame
    Returns a dictionary where-
    key is animal_id, value in Pandas DataFrame for that 'animal_id'
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
            Distance=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            Average_Speed=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            Average_Acceleration=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            Direction=data)

    return data_animal_id_groups


def filter_dataframe(data, frm, to):
    """
    A function to filter the dataset, which is the first
    argument to the function using 'frm' and 'to' as the
    second and third arguments.
    Please note that both 'frm' and 'to' are included in
    the returned filtered Pandas Data frame.

    Returns a filtered Pandas Data frame according to 'frm'
    and 'to' arguments
    """

    return data.loc[(data['time'] >= frm) & (data['time'] < to), :]


def replace_parts_animal_movement(data_groups, animal_id, time_array, replacement_value_x, replacement_value_y):
    """
    Replace subsets (segments) of animal movement based on some indices e.g. time
    This function can be used to remove outliers

    Input:
    1.) First argument is a Python 3 dictionary whose key is 'animal_id'and value is
    Pandas DataFrame for that 'animal_id'
    2.) Second argument is 'animal_id' whose movements have to replaced
    3.) Third argument is an array of time indices whose movements have to replaced
    4.) Fourth argument is the value which will be replaced for all values contained
    in 'time_array' for 'x' attribute
    5.) Fifth argument is the value which will be replaced for all values contained
    in 'time_array' for 'y' attribute

    Returns:
    Modified Python 3 dictionary which was passed as first argument to it

    An example usage-

    data = csv_to_pandas(path)

    data_groups = group_animals(data)

    arr_index = np.array([10, 20, 200, 20000, 40000, 43200])

    replaced_data_groups = replace_parts_animal_movement(data_groups, 811, arr_index, 100, 90)
        """
    data_groups[animal_id].loc[time_array, 'x'] = replacement_value_x
    data_groups[animal_id].loc[time_array, 'y'] = replacement_value_y

    return data_groups
