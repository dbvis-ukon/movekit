

import pandas as pd
import numpy as np
import time
from scipy.spatial import distance


def medoid_computation(data):
	'''
	Calculates the data point (animal_id) closest to
	center/centroid/medoid for a time step
    Uses group by on 'time' attribute

    Input:      Expects a Pandas CSV input parameter containing the dataset
    Returns:    Pandas DataFrame containing computed medoids & centroids
    '''

	# Create Python dictionary to hold final medoid computation-
	data_d = {
			'time': [0 for x in range(data.shape[0])],
			'x_coordinate_centroid': [0 for x in range(data.shape[0])],
			'y_coordinate_centroid': [0 for x in range(data.shape[0])], 
			'medoid': [0 for x in range(data.shape[0])]
			}                                   

	# Create Pandas Dataframe using dict from above-
	medoid_data = pd.DataFrame(data_d)


	# Group by 'time'-
	data_time = data.groupby('time')

	# Dictionary to hold grouped data by 'time' attribute-
	data_groups_time = {}

	for aid in data_time.groups.keys():
		data_groups_time[aid] = data_time.get_group(aid)

	# Reset index-
	for aid in data_time.groups.keys():
		data_groups_time[aid].reset_index(drop = True, inplace = True)


	# NOTE:
	# Each group has only five entries
	# Each group has dimension- (5, 4)


	# Add 3 additional columns to each group-
	for aid in data_groups_time.keys():
		data_l = [0 for x in range(data_groups_time[aid].shape[0])]

		data_groups_time[aid] = data_groups_time[aid].assign(x_centroid = data_l)
		data_groups_time[aid] = data_groups_time[aid].assign(y_centroid = data_l)
		data_groups_time[aid] = data_groups_time[aid].assign(medoid = data_l)
		data_groups_time[aid] = data_groups_time[aid].assign(distance_to_centroid = data_l)


	for tid in data_groups_time.keys():
		# Calculate centroid coordinates (x, y)-
		x_mean = np.around(np.mean(data_groups_time[tid]['x']), 3)
		y_mean = np.around(np.mean(data_groups_time[tid]['y']), 3)
		centroid = np.asarray([x_mean, y_mean])

		data_groups_time[tid] = data_groups_time[tid].assign(x_centroid = x_mean)
		data_groups_time[tid] = data_groups_time[tid].assign(y_centroid = y_mean)

		# Squared distance of each 'x' coordinate to 'centroid'-
		x_temp = (data_groups_time[tid].loc[:, 'x'] - x_mean) ** 2

		# Squared distance of each 'y' coordinate to 'centroid'-
		y_temp = (data_groups_time[tid].loc[:, 'y'] - y_mean) ** 2

		# Distance of each point from centroid-
		dist = np.sqrt(x_temp + y_temp)

		# Assign computed distances to 'distance_to_centroid' attribute-
		data_groups_time[tid] = data_groups_time[tid].assign(distance_to_centroid = np.around(dist, decimals = 3))

		# Find 'animal_id' nearest to centroid for this group-
		pos = np.argmin(data_groups_time[tid]['distance_to_centroid'].values)
		nearest = data_groups_time[tid].loc[pos, 'animal_id']

		# Assign 'medoid' for this group-
		data_groups_time[tid] = data_groups_time[tid].assign(medoid = nearest)

		medoid_data.loc[tid, 'time'] = tid
		medoid_data.loc[tid, 'x_coordinate_centroid'] = x_mean
		medoid_data.loc[tid, 'y_coordinate_centroid'] = y_mean
		medoid_data.loc[tid, 'medoid'] = nearest

		# Drop index 0-
		medoid_data.drop(medoid_data.index[0], inplace=True)


	# return medoid_data, data_groups_time
	return medoid_data


# dist_scipy = distance.euclidean(a, b)
# dist_np = np.linalg.norm(a-b, ord = 2)


# Example usage-

# Read in CSV file-
data = pd.read_csv("fish-5.csv")

# Sort file by 'time' attribute-
data.sort_values('time', ascending=True, inplace = True)


# start_time = time.time()

# Compute medoid of data-
medoid_data = medoid_computation(data)

# end_time = time.time()
# print("\nTotal time taken for medoid computation = {0:.4f} seconds\n".format(end_time - start_time))
# Total time taken for medoid computation = 824.1987 seconds
# Total time taken for medoid computation = 899.4437 seconds


# Optional-
# medoid_data.to_csv("medoid_computation.csv", index=False)