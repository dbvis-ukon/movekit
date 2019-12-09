import pandas as pd
import numpy as np
import math


def resample_systematic(data_groups, downsample_size):

	"""
    Resample the movement data of each animal - by downsampling at fixed time
    intervals. This is done to reduce the resolution of dataset. This function
	does this by systematically choosing samples from each animal.

    Input:
    1.) data_groups is a Python 3 dictionary containing as key 'animal_id' and
    it's value is Pandas DataFrame pertaining to that 'animal_id'
    2.) downsample_size is the sample size to which each animal has to be
    downsampled to

    Returns:
    Modified 'data_groups' Python 3 dictionary to 'downsample_size'
    """

    # Get first key-
    first = list(data_groups.keys())[0]

    # size of each animal's group-
    size = data_groups[first].shape[0]

	step_size = math.floor(size / downsample_size)
    arr_index = []
    l = list(range(size))
    arr_index = l[0:(step_size * downsample_size):step_size]

    # Convert list to numpy array-
    arr_index = np.asarray(arr_index)

    # Modified 'data_groups' downsampled Python 3 dictionary-
    data_groups_downsampled = {}

    for aid in data_groups.keys():
        data_groups_downsampled[aid] = data_groups[aid].loc[arr_index, :]

    return data_groups_downsampled


def resample_random(data_groups, downsample_size):

	"""
    Resample the movement data of each animal - by downsampling at random time
    intervals. This is done to reduce the resolution of dataset. This function
	does this by randomly choosing samples from each animal

    Input:
    1.) data_groups is a Python 3 dictionary containing as key 'animal_id' and
    it's value is Pandas DataFrame pertaining to that 'animal_id'
    2.) downsample_size is the sample size to which each animal has to be
    downsampled to

    Returns:
    Modified 'data_groups' Python 3 dictionary to 'downsample_size'

	# An example usage-
		data = csv_to_pandas("/home/arjun/University_of_Konstanz/Hiwi/Work/Movement_Patterns/fish-5.csv")
		data_groups = group_animals(data)
		modified_data_groups = resample_systematic(data_groups, 200)
		print("\nPrinting dimensions of downsampled systematic dataset Python 3 dict:\n")
		for aid in modified_data_groups.keys():
			print("animal_id = {0} & shape = {1}".format(aid, modified_data_groups[aid].shape))

		modified_data_groups_random = resample_random(data_groups, 1000)
		print("\nPrinting dimensions of downsampled random dataset Python 3 dict:\n")
		for aid in modified_data_groups_random.keys():
			print("animal_id = {0} & shape = {1}".format(aid, modified_data_groups_random[aid].shape))
    """

    # Get first key-
    first = list(data_groups.keys())[0]

    # size of each animal's group-
    size = data_groups[first].shape[0]

    # Random index (numpy.ndarray)-
    ix_random = np.random.randint(0, size, downsample_size)

    # Modified 'data_groups' downsampled Python 3 dictionary-
    data_groups_downsampled = {}

    for aid in data_groups.keys():
        data_groups_downsampled[aid] = data_groups[aid].loc[ix_random, :]

    return data_groups_downsampled
