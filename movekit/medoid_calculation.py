

import pandas as pd
import numpy as np
import math


def calculate_centroid(data_groups):
    """
    Computes the distance of each animal from the centroid of the group
	
    Input: Expects a dictionary containing-
    animal_id as key and Pandas Data Frame for that animal_id as value
	
    Use Pandas group by to create such a Python 3 dictionary
	
    Returns: A modified Pandas Data Frame containing 'distance_centroid'
    attribute
    """

    for group in data_groups.keys():
        x_mean = data_groups[group]['x'].mean()
        y_mean = data_groups[group]['y'].mean()
        # print("\nGroup = {0}, x_mean = {1:.3f} and y_mean = {2:.3f}".format(group, x_mean, y_mean))

        x = np.asarray(data_groups[group]['x'])
        y = np.asarray(data_groups[group]['y'])

        x_temp = (x - x_mean)**2
        y_temp = (y - y_mean)**2
        dist = np.sqrt(x_temp + y_temp)

        data_groups[group] = data_groups[group].assign(
            distance_to_centroid=np.around(dist, decimals=3))

    # Show 'distance_x' attribute less than zero-
    # data_groups[905].loc[data_groups[905]['distance_x'] < 0, ]

    # Concatenate different groups into one Pandas DataFrame-
    result = pd.concat(data_groups[aid] for aid in data_groups.keys())

    # Reset indices-
    result.reset_index(drop=True, inplace=True)

    # Write file to HDD (optional)-
    # result.to_csv("fish-5_centroid.csv", index=False)

    return result
    # return data_groups


def medoid_computation(data):
    '''
    Calculates the data point (animal_id) closest to center/centroid
    for a time step
    Uses group by on 'time' attribute

    Input:      Expects a Pandas CSV input parameter containing the dataset
    Returns:    Python 3 dictionary having as key, 'time' and as values,
                Pandas DataFrame associated with it
    '''


    # Group by 'time'-
    # Group according to 'animal_id' attribute-
    data_time = data.groupby('time')

    # Dictionary to hold grouped data by 'time' attribute-
    data_groups_time = {}

    for aid in data_time.groups.keys():
        data_groups_time[aid] = data_time.get_group(aid)

    # Reset index-
    for aid in data_time.groups.keys():
        data_groups_time[aid].reset_index(drop = True, inplace = True)


    # Each group has the dimension-
    # data_groups_time[10].shape
    # (5, 4)

    for tid in data_groups_time.keys():

        # Compute centroid for each time step-
        x_mean = data_groups_time[tid]['x'].mean()
        y_mean = data_groups_time[tid]['y'].mean()

        # Centroid of this group-
        # x_mean, y_mean

        # print("\nCentroid of this group: x = {0:.4f} & y: {1:.4f}\n".format(x_mean, y_mean))

        x = np.asarray(data_groups_time[tid]['x'])
        y = np.asarray(data_groups_time[tid]['y'])

        x_temp = (x - x_mean) ** 2
        y_temp = (y - y_mean) ** 2

        dist = np.sqrt(x_temp + y_temp)

        data_groups_time[tid] = data_groups_time[tid].assign(distance_to_centroid = np.around(dist, decimals = 3))

        # Find 'animal_id' nearest to centroid for this group-
        pos = np.argmin(data_groups_time[tid]['distance_to_centroid'].values)
        nearest = data_groups_time[tid].loc[pos, 'animal_id']

        # Assign 'medoid' for this group-
        data_groups_time[tid] = data_groups_time[tid].assign(medoid = nearest)


    # Concatenate different groups into one Pandas DataFrame-
    result = pd.concat(data_groups_time[aid] for aid in data_groups_time.keys())

    # Reset indices-
    result.reset_index(drop=True, inplace=True)
 
    # return data_groups_time
    return result


# Example usage-

# Read in data-
# data = pd.read_csv("fish-5.csv")

# Sort values by 'time' attribute-
# data.sort_values("time", ascending=True, inplace = True)


"""
# Group according to 'animal_id' attribute-
data_animals = data.groupby('animal_id')

# A dict object to store different groups created above as-
# animal_id: pandas dataframe for that animal_id
data_groups = {}

# Add different 'animal_id' in dict-
for aid in data_animals.groups.keys():
	data_groups[aid] = data_animals.get_group(aid)

# Reset index-
for aid in data_animals.groups.keys():
	data_groups[aid].reset_index(drop = True, inplace = True)


# Compute centroid-
# data_centroid = calculate_centroid(data_groups)

"""


# Compute medoid-
# data_medoid = medoid_computation(data)

# Check for rows where 'medod' != 312-
# data_medoid[data_medoid['medoid'] != 312]

# Optional (Save to HDD)-
# data_medoid.to_csv("medoid_computation.csv", index=False)


