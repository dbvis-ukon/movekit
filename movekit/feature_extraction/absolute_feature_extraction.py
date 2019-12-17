

import numpy as np
import pandas as pd


# Read in CSV file-
def csv_to_pandas(path_to_file):
	'''
	A function to read CSV file into a Pandas DataFrame-
	Expects complete path/relative path to CSV file along with file name
	'''
	# data = pd.read_csv("fish-5.csv")

	try:

		if path_to_file[-3:] == 'csv':
			data = pd.read_csv(path_to_file)
		else:
			data = pd.read_csv(path_to_file + '.csv')

			# Check if 'time' attribute is integer-
			if is_numeric_dtype(data['time']):
				data.sort_values('time', ascending = True, inplace = True)
			# Check if 'time' attribute is string-
			elif is_string_dtype(data['time']):
				data['time'] = pd.to_datetime(data['time'])
				data.sort_values('time', ascending = True, inplace = True)

		return data

	except FileNotFoundError:
		print("Your file below could not be found. Please check path and/or file name and try again.\nPath given: {0}\n\n".format(path_to_file))


def grouping_data(processed_data):
	'''
	A function to group all values for each 'animal_id'
	Input is 'processed_data' which is processed Pandas DataFrame
	Returns a dictionary where-
	key is animal_id, value in Pandas DataFrame for that 'animal_id'
	'''

	# A dictionary object to hold all groups obtained using group by-
	# Apply grouping using 'animal_id' attribute-
	data_animal_id = processed_data.groupby('animal_id')

	# A dictionary object to hold all groups obtained using group by-
	data_animal_id_groups = {}


	# Get each animal_id's data from grouping performed-
	for animal_id in data_animal_id.groups.keys():
		data_animal_id_groups[animal_id] = data_animal_id.get_group(animal_id)

	# To reset index for each group-
	for animal_id in data_animal_id_groups.keys():
		data_animal_id_groups[animal_id].reset_index(drop=True, inplace=True)


	# Add additional attributes/columns to each groups-
	for aid in data_animal_id_groups.keys():
		data = [0 for x in range(data_animal_id_groups[aid].shape[0])]

		data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(distance=data)
		data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(average_speed=data)
		data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(average_acceleration=data)
		data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(positive_acceleration=data)
		data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(direction=data)


	return data_animal_id_groups


def compute_distance_and_direction(data_animal_id_groups):
	'''
	Function to calculate metric distance and direction attributes-

	Calculate the metric distance between two consecutive time frames/time stamps
	for each moving entity (in this case, fish)

	Accepts a Python 3 dictionary
	Returns a Python 3 dictionary containing computed 'distance'
	and 'direction' attributes
	'''

	# Compute 'direction' for 'animal_id' groups-
	for aid in data_animal_id_groups.keys():
		data_animal_id_groups[aid]['direction'] = np.rad2deg(
			np.arctan2((
				data_animal_id_groups[aid]['y'] -
				data_animal_id_groups[aid]['y'].shift(periods = 1)),
			(data_animal_id_groups[aid]['x'] -
				data_animal_id_groups[aid]['x'].shift(periods = 1))))


	# Compute 'distance' for 'animal_id' groups-
	for aid in data_animal_id_groups.keys():
		print("\nComputing 'distance' attribute for Animal ID = {0}\n".format(aid))

		p1 = data_animal_id_groups[aid].loc[:, ['x', 'y']]
		p2 = data_animal_id_groups[aid].loc[:, ['x', 'y']].shift(periods = 1)
		p2.iloc[0,:] = [0.0, 0.0]

		data_animal_id_groups[aid]['distance'] = ((p1 - p2) ** 2).sum(axis = 1) ** 0.5


	# Reset first entry for each 'animal_id' to zero-
	for aid in data_animal_id_groups.keys():
		data_animal_id_groups[aid].loc[0, 'distance'] = 0.0


	return data_animal_id_groups


def compute_average_speed(data_animal_id_groups, fps):
	'''
	Function to compute average speed of an animal based on fps
	(frames per second) parameter. Calculate the average speed of a mover,
	based on the pandas dataframe and a frames per second (fps) parameter

	Formula used-
	Average Speed = Total Distance Travelled / Total Time taken

	Input- Python dict and fps
	Returns- Python dict
	'''
	for aid in data_animal_id_groups.keys():
		print("\nComputing 'average_speed' attribute for animal id = {0}\n".format(aid))
		data_animal_id_groups[aid]['average_speed'] = data_animal_id_groups[aid]['distance'].rolling(window = fps, win_type = None).sum() / fps

	return data_animal_id_groups


def compute_average_acceleration(data_animal_id_groups, fps):
	'''
	A function to compute average acceleration of an animal based on fps
	(frames per second) parameter.

	Formulas used are-
	Average Acceleration = (Final Speed - Initial Speed) / Total Time Taken

	Input- Python 3 dict and fps
	Returns- Pandas DataFrame containing computations
	'''
	for aid in data_animal_id_groups.keys():
		print("\nComputing 'average_acceleration' attribute for animal ID = {0}\n".format(aid))

		a = data_animal_id_groups[aid]['average_speed']
		b = data_animal_id_groups[aid]['average_speed'].shift(periods = 1)

		data_animal_id_groups[aid]['average_acceleration'] = (a - b) / fps


	# Concatenate all Pandas DataFrame into one-
	result = pd.concat(data_animal_id_groups[aid] for aid in data_animal_id_groups.keys())

	# Reset index-
	result.reset_index(drop=True, inplace=True)

	return result




# Get absolute path and file name with extension from user-
path_to_file = input("Enter path to data file: ")

# /home/arjun/University_of_Konstanz/Hiwi/Work/Movement_Patterns/fish-5.csv

# Get CSV data file as Pandas DataFrame-
data = csv_to_pandas(path_to_file)


# Group given data according to 'animal_id' attribute-
data_animal_id_groups = grouping_data(data)

# Compute distance and direction attributes-
distance_and_direction_data = compute_distance_and_direction(data_animal_id_groups)


# Get fps input from user-
fps = int(input("\nEnter frames per second (fps) parameter: "))

# Compute 'average_speed' attribute using 'fps' attribute-
avg_speed_data = compute_average_speed(distance_and_direction_data, fps)

# Compute 'average_acceleration' attribute using 'fps' attribute-
avg_acc_data = compute_average_acceleration(avg_speed_data, fps)

# NOTE: 'avg_acc_data' is Panda DataFrame and NOT a Python 3 dict!

# Sanity check-
type(avg_acc_data)
# pandas.core.frame.DataFrame




# Optional- Write computed CSV to HDD-
# result.to_csv("fish-5_{0}_fps_computed_attributes.csv".format(fps), index=False)
