"""
  Preprocess the data frame 
  Author: Arjun Majumdar, Eren Cakmak
  Created: July, 2019
"""

import pandas as pd


def data_preprocessing(data):
    '''
    A function to perform data preprocessing and interpolation
    Expects 'data' as input which is the Pandas DataFrame to be processed
    '''

    print(
        "\nThe dimensions/shape of the raw data file is: {0}\n".format(data.shape))
    print("\nNumber of unique animals in raw data are: {0}\n".format(
        data['animal_id'].nunique()))

    print("\nNumber of rows in data having missing values for 'time' attribute are = {0}\n".format(
        len(list(data[data['time'].isnull()].index))))
    print("\nNumber of rows in data having missing values for 'animal_id' attribute are = {0}\n".format(
        len(list(data[data['animal_id'].isnull()].index))))
    print("\nRows having missing values for 'time' and 'animal_id' will be deleted.\n")

    # Check if 'time' attribute has missing values
    # If yes, delete all rows having missing values
    if data['time'].isnull().values.any():
        data = data[pd.notnull(data['time'])]

    # Check if 'animal_id' attribute has missing values
    # If yes, delete all rows having missing values
    if data['animal_id'].isnull().values.any():
        data = data[pd.notnull(data['animal_id'])]

    # Find duplicate rows based on 'time' & 'animal_id' attributes-
    duplicate_rows = data[data.duplicated(subset=['x', 'y'], keep='first')]

    # Get indices for duplicate rows-
    # duplicate_rows.index
    # OR-
    # list(duplicate_rows.index)

    print("\nNumber of duplicate rows in data for 'x' & 'y' attributes are = {0}\n".format(
        len(list(duplicate_rows.index))))
    print("\nDuplicate rows for 'x' & 'y' attributes will be removed.\n")

    # Remove the duplicated rows found above-
    data.drop(axis=0, index=list(duplicate_rows.index), inplace=True)

    # Return processed data-
    return data


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
