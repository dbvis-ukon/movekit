import pandas as pd
import numpy as np
import math


def split_trajectories(data_groups, segment=1):

	"""
    Split the trajectory of a single animal into several intervals (segments)
    according to some specific criterion.

    Splitting may be interesting for example to detect different properties in
    time intervals. E.g. split into segments of 1 minute

    Accepts: Python 3 dictionary containing 'animal_id' as key, and it's Pandas
    Data Frame as value AND
    segments: an interval according to which the animals will be split into several
    Pandas Data Frames

    Returns: Nothing. All segmented Pandas Data Frames are saved to HDD

    # An example usage-
    	data = csv_to_pandas("/home/arjun/University_of_Konstanz/Hiwi/Work/Movement_Patterns/fish-5.csv")
    	data_groups = group_animals(data)
    	split_trajectories(data_groups, segment = 3)
    """

    # Method-1:
    # df1, df2 = data_groups[312].iloc[:10, :], data_groups[312].iloc[10:20, :]

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
            groups['group_' + str(aid) + '_df' +
                   str(l + 1)] = data_groups[aid].iloc[beg:end, :]
            beg, end = end, end + segment_size

    for k in groups.keys():
        groups[k].to_csv(k + '.csv', index=False)

    return None


def split_trajectories_fuzzy_segmentation(data_groups, segment=1,
										  fuzzy_segment=2):

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

    # An example usage-
    	data = csv_to_pandas("/home/arjun/University_of_Konstanz/Hiwi/Work/Movement_Patterns/fish-5.csv")
    	data_groups = group_animals(data)
    	split_trajectories_fuzzy_segmentation(data_groups, segment = 5, fuzzy_segment = 5)
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
            groups['group_' + str(aid) + '_df' +
                   str(l + 1)] = data_groups[aid].iloc[beg:end, :]
            beg, end = end - fuzzy_segment, end + segment_size + fuzzy_segment

    for k in groups.keys():
        groups[k].to_csv(k + '.csv', index=False)

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
