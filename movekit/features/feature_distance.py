

import pandas as pd
import numpy as np
import math


def compute_distance_and_direction(data_animal_id_groups):
	'''
	Function to calculate the sum of metric distance travelled by an animal.
	Also calculates the maximum distance travelled by an animal.

	Input: Accepts a Python 3 dictionary containing animal_id as key and its
	Pandas Data Frame as value

	Returns: A Python 3 dictionary containing two nested dictionaries computing-
	sum of total metric distance travelled by each animal
	maximum distance travelled by each animal
	'''

	distance_metric_sum = {}
	distance_max = {}

	for aid in data_animal_id_groups.keys():
		# print("\nComputing Distance & Direction for Animal ID = {0}\n".format(aid))
		distance_sum = 0
		max_distance = 0

		# for i in range(1, animal_id.shape[0] - 1):
		for i in range(1, data_animal_id_groups[aid].shape[0] - 1):
			# print("Current i = ", i)

			x1 = data_animal_id_groups[aid].iloc[i, 2]
			y1 = data_animal_id_groups[aid].iloc[i, 3]
			x2 = data_animal_id_groups[aid].iloc[i + 1, 2]
			y2 = data_animal_id_groups[aid].iloc[i + 1, 3]
	
			# Compute distance between 2 points-
			distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))

			if distance > max_distance:
				max_distance = distance

			# Insert computed distance to column/attribute 'Distance'-
			# animal_id.loc[i, 'Distance'] = distance
			# data_animal_id_groups[aid].loc[i, 'Distance'] = distance
			distance_sum += distance

		distance_metric_sum[aid] = distance_sum
		distance_max[aid] = max_distance

		distance = {'sum_of_distance': distance_metric_sum, 'maximum_distance': distance_max}


	return distance


# Example usage:

# Read in data-
data = pd.read_csv("/home/arjun/University_of_Konstanz/Hiwi/Work/Movement_Patterns/fish-5.csv")

# Sorting by 'time' attribute-
data.sort_values('time', ascending = True, inplace = True)

# Group according to 'animal_id' attribute-
data_animals = data.groupby('animal_id')

# A dict object to store different groups created above-
data_groups = {}

# Add different 'animal_id' in dict-
for aid in data_animals.groups.keys():
	data_groups[aid] = data_animals.get_group(aid)

# Reset index-
for aid in data_animals.groups.keys():
	data_groups[aid].reset_index(drop = True, inplace = True)


distance = compute_distance_and_direction(data_groups)

print("\nThe sum of distance travelled by each animal id is:")
for aid in distance['sum_of_distance'].keys():
	print("Animal ID: {0} & total metric distance travelled = {1:.2f}".format(aid, distance['sum_of_distance'][aid]))


print("\nThe maximum distance travelled by each animal id is:")
for aid in distance['maximum_distance'].keys():
	print("Animal ID: {0} & total metric distance travelled = {1:.2f}".format(aid, distance['maximum_distance'][aid]))



