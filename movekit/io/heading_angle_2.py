

import pandas as pd
import numpy as np
import math


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

        # Change attribute/column names to lowe case-
        data.columns = map(str.lower, data.columns)

        # Check if all required attribute/columns are present-
        if ['time', 'animal_id', 'x', 'y', 'heading_angle'] == data.columns.tolist():
            # Check if 'time' attribute is integer-
            if np.issubdtype(data['time'].dtype, np.number):
                data.sort_values('time', ascending = True, inplace = True)

            # Check if 'time' attribute is string-
            elif np.issubdtype(data['time'].dtype, np.number) == False:
                data['time'] = pd.to_datetime(data['time'])
                data.sort_values('time', ascending = True, inplace = True)

            return data

        else:
            print("\n'heading_angle' attribute is NOT present in the CSV file. Proceeding without the column!\n")

            # Check if 'time' attribute is numeric-
            if np.issubdtype(data['time'].dtype, np.number):
                data.sort_values('time', ascending = True, inplace = True)

            # Check if 'time' attribute is non-numeric-
            elif np.issubdtype(data['time'].dtype, np.number) == False:
                data['time'] = pd.to_datetime(data['time'])
                data.sort_values('time', ascending = True, inplace = True)

            return data

    except FileNotFoundError:
        print("\nThe given path to file and/or the file below, could NOT be found:\n{0}\nPlease try again!\n".format(path_to_file))


# Example usage-

# Read in CSV file having 'heading_angle' attribute in it-
# data = input_heading_angle("/path_to_file/fish-5.csv")

# The following functions are in 'absolute.py' in 'feature_extraction' folder-

# Group data into a Python 3 dictionary according to 'animal_id' attribute-
# grouped_data = grouping_data(data)

# Compute distance for Python 3 dictionary-
# distance_data = compute_distance(grouped_data)


"""
# You can save the computed Python 3 dictionary for later use
# Example-

import pickle

# Dump Python 3 dictionary as pickle-
with open("computing_distance_heading_angle.pickle", "wb") as pickle_dump:
    pickle.dump(distance_data, pickle_dump)


# Load Python 3 dictionary from pickle file-
with open("computing_distance_heading_angle.pickle", "rb") as pickle_input:
    distance_data = pickle.load(pickle_input)


# Sanity check-
distance_data.keys()                                                    
# dict_keys([312, 511, 607, 811, 905])
"""


