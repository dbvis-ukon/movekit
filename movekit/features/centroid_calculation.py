

import pandas as pd
import numpy as np


"""
Example code:

# Read in Pandas Data Frame-
data = pd.read_csv("fish-5.csv")

data.columns.tolist()
# ['time', 'animal_id', 'x', 'y']


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


# data_groups.keys()
"""


def calculate_centroid(data_groups):
	"""
	Computes the distance of each animal from the centroid of the group
	
	Input: Expects a dictionary containing:
	animal_id as key and Pandas Data Frame for that animal_id as value
	
	Use Pandas group by to create such a Python 3 dictionary
	
	Returns: A Pandas Data Frame
	"""

	for group in data_groups.keys():
		x_mean = data_groups[group]['x'].mean()
		y_mean = data_groups[group]['y'].mean()
		# print("\nGroup = {0}, x_mean = {1:.3f} and y_mean = {2:.3f}".format(group, x_mean, y_mean))

		data_groups[group] = data_groups[group].assign(distance_x = np.around(data_groups[group]['x'] - x_mean, decimals = 3))
		data_groups[group] = data_groups[group].assign(distance_y = np.around(data_groups[group]['y'] - y_mean, decimals = 3))


	# Show 'distance_x' attribute less than zero-
	# data_groups[905].loc[data_groups[905]['distance_x'] < 0, ]


	# Concatenate different groups into one Pandas DataFrame-
	result = pd.concat(data_groups[aid] for aid in data_groups.keys())

	# Reset indices-
	result.reset_index(drop = True, inplace = True)

	# Write file to HDD (optional)-
	# result.to_csv("fish-5_centroid.csv", index=False)

	return result


# Example usage-
# data_processed = calculate_centroid(data_groups)


