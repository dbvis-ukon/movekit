

import pandas as pd, numpy as np


def csv_to_pandas(path_to_file):
	'''
	A function to read CSV file into a Pandas DataFrame-
	Expects complete path/relative path to CSV file along with file name

	Input: Accepts absolute path to CSV file
	Output: Returns Pandas DataFrame sorted by 'time' attribute in ascending
	order
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


def group_animals(data):
	"""
	Function to group animals in 'data' Pandas Data Frame according to
	'animal_id' attribute

	Accepts: Pandas Data Frame sorted according to 'time' attribute in ascending
	order
	Returns: Python 3 dictionary containing 'animal_id' as key, and it's Pandas
	Data Frame as value
	"""
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

	return data_groups


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
	"""
	data_groups[animal_id].loc[time_array, 'x'] = replacement_value_x
	data_groups[animal_id].loc[time_array, 'y'] = replacement_value_y

	return data_groups


"""
# An example usage-

data = csv_to_pandas("/home/arjun/University_of_Konstanz/Hiwi/Work/Movement_Patterns/fish-5.csv")

data_groups = group_animals(data)

arr_index = np.array([10, 20, 200, 20000, 40000, 43200])

replaced_data_groups = replace_parts_animal_movement(data_groups, 811, arr_index, 100, 90)
"""

