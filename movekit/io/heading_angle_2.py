

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


def grouping_data(processed_data):
	'''
	A function to group all values for each 'animal_id'

        Input: 'processed_data' which is a processed Pandas DataFrame
        Returns: Python 3 dictionary where-
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
		data_animal_id_groups[animal_id].reset_index(drop = True, inplace = True)

	# Add additional attributes/columns to each groups-
	for aid in data_animal_id_groups.keys():
		data = [0 for x in range(data_animal_id_groups[aid].shape[0])]
    
		data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(Distance = data)
		data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(Average_Speed = data)
		data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(Average_Acceleration = data)
		# data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(Direction = data)

	return data_animal_id_groups


def compute_distance(data_animal_id_groups):
    '''
    Function to calculate metric distance between two consecutive time frames/time stamps
    for each moving entity (in this case, fish)

    Input: Python 3 dictionary having as key the animal_id & as value, Pandas DataFrame
    associated with it
    Returns: Processed Python 3 dictionary provided as input to function having computed
    distance in it 
    '''

    # start_time = time.time()

    for aid in data_animal_id_groups.keys():
        print("\nComputing Distance for Animal ID = {0}\n".format(aid))

        for i in range(1, data_animal_id_groups[aid].shape[0] - 1):
            x1 = data_animal_id_groups[aid].iloc[i, 2]
            y1 = data_animal_id_groups[aid].iloc[i, 3]
            x2 = data_animal_id_groups[aid].iloc[i + 1, 2]
            y2 = data_animal_id_groups[aid].iloc[i + 1, 3]

            # Compute distance between 2 points-
            distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))

            """
            # Compute the direction in DEGREES-
            direction = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
            if matsh.isnan(direction):
            data_animal_id_groups[aid].loc[i, 'Direction'] = 0
	    else:
                data_animal_id_groups[aid].loc[i, 'Direction'] = direction
            """

            # Insert computed distance to column/attribute 'Distance'-
            # animal_id.loc[i, 'Distance'] = distance
            data_animal_id_groups[aid].loc[i, 'Distance'] = distance

    # end_time = time.time()
    # print("\nTime taken to create distance & direction data = {0:.4f} seconds\n\n".format(end_time - start_time))
    # Time taken to create distance & direction data = 1013.1692 seconds

    return data_animal_id_groups




# Example usage-

# Read in CSV file having 'heading_angle' attribute in it-
data = input_heading_angle("/home/arjun/University_of_Konstanz/Hiwi/Work/Movement_Patterns/fish-5.csv")

# Group data into a Python 3 dictionary according to 'animal_id' attribute-
grouped_data = grouping_data(data)

# Compute distance for Python 3 dictionary-
distance_data = compute_distance(grouped_data)


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


