

import pandas as pd
import numpy as np
import math


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


def split_trajectories_fuzzy_segmentation(data_groups, segment = 1, fuzzy_segment = 2):
	"""
	Split the trajectory of a single animal into several intervals (segments)
	according to some specific criterion.

	Splitting may be interesting for example to detect different properties in
	time intervals. E.g. split into segments of 1 minute

	Th fuzzy temporal segmentation is if the dataset is segmented into 10 min
	intervals, add a window of 2 minutes that overlaps on either side of the
	segments

	Accepts: Python 3 dictionary containing 'animal_id' as key, and it's Pandas
	Data Frame as value AND
	segments: an interval according to which the animals will be split into several
	Pandas Data Frames
	fuzzy_segment: an interval which will overlap on either side of the segments
	
	Returns: Nothing. All segmented Pandas Data Frames are saved to HDD
	"""

	# Get first key-
	first = list(data_groups.keys())[0]

	size = data_groups[first].shape[0]
	segment_size = math.floor(size / segment)
	
	
	groups = {}

	for aid in data_groups.keys():
		beg, end = 0, segment_size
		# groups['group_' + str(aid)] = data_groups[aid]
		for l in range(segment):
			# groups['group_' + str(aid)]['df' + str(l + 1)] = data_groups[aid].iloc[beg: end, :]
			groups['group_' + str(aid) + '_df' + str(l + 1)] = data_groups[aid].iloc[beg: end, :]
			beg, end = end - fuzzy_segment, end + segment_size + fuzzy_segment

	for k in groups.keys():
		groups[k].to_csv(k + '.csv', index = False)

	"""
	# Code to check indexing for fuzzy segmentation-
	groups = {}

	for aid in data_groups.keys():
		beg, end = 0, segment_size + fuzzy_segment
		for l in range(segment):
			print("\nCurrent 'beg' and 'end' are:")
			print("beg = {0} and end = {1}".format(beg, end))
			beg, end = end - fuzzy_segment, end + segment_size + fuzzy_segment
	"""

	return None


"""
# An example usage-

data = csv_to_pandas("/home/arjun/University_of_Konstanz/Hiwi/Work/Movement_Patterns/fish-5.csv")

data_groups = group_animals(data)

split_trajectories_fuzzy_segmentation(data_groups, segment = 5, fuzzy_segment = 5)
"""






