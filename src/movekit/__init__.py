

import numpy as np
import pandas as pd
import time
from scipy.spatial import distance
import matplotlib.pyplot as plt
from tsfresh import select_features
from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pandas.errors import EmptyDataError




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

	Use output of grouping_data() function to this function.

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

	Use output of compute_distance_and_direction() function to this function.

	Input- Python dict and fps
	Returns- Python dict
	'''
	for aid in data_animal_id_groups.keys():
		print("\nComputing 'average_speed' attribute for animal id = {0}\n".format(aid))
		data_animal_id_groups[aid]['average_speed'] = data_animal_id_groups[aid] \
		['distance'].rolling(window = fps, win_type = None).sum() / fps

	return data_animal_id_groups


def compute_average_acceleration(data_animal_id_groups, fps):
	'''
	A function to compute average acceleration of an animal based on fps
	(frames per second) parameter.

	Formulas used are-
	Average Acceleration = (Final Speed - Initial Speed) / Total Time Taken

	Use output of compute_average_speed() function to this function.

	Input- Python 3 dict and fps
	Returns- Pandas DataFrame containing computations
	'''
	for aid in data_animal_id_groups.keys():
		print("\nComputing 'average_acceleration' attribute for animal ID = {0}\n". \
			format(aid))

		a = data_animal_id_groups[aid]['average_speed']
		b = data_animal_id_groups[aid]['average_speed'].shift(periods = 1)

		data_animal_id_groups[aid]['average_acceleration'] = (a - b) / fps


	# Concatenate all Pandas DataFrame into one-
	result = pd.concat(data_animal_id_groups[aid] for aid in data_animal_id_groups.keys())

	# Reset index-
	result.reset_index(drop=True, inplace=True)

	return result


def compute_absolute_features(data_animal_id_groups, fps=10, stop_threshold=0.5):
	'''
	Calculate absolute features for the input data animal group.

	Input- Python 3 dictionary, fps (frames per second) and stopping threshold
	Returns- Pandas Python 3 dictionary
	'''

	direction_distance_data = compute_distance_and_direction(data_animal_id_groups)

	avg_speed_data = compute_average_speed(direction_distance_data, fps)

	avg_acceleration_data = compute_average_acceleration(avg_speed_data, fps)

	stop_data = computing_stops(avg_acceleration_data, stop_threshold)

	return stop_data


def computing_stops(data_animal_id_groups, threshold_speed):
    '''
    Calculate absolute feature called 'Stopped' where the value is 'yes'
    if 'Average_Speed' <= threshold_speed and 'no' otherwise

    Input- Python 3 dictionary and threshold speed
	Returns- Python 3 dictionary
    '''
    data_animal_id_groups['stopped'] = np.where(
        data_animal_id_groups['average_speed'] <= threshold_speed, 1, 0)

    print(
        "\nNumber of movers stopped according to threshold speed = {0} is {1}".
        format(threshold_speed, data_animal_id_groups['stopped'].eq(1).sum()))

    print(
        "Number of movers moving according to threshold speed = {0} is {1}\n".
        format(threshold_speed, data_animal_id_groups['stopped'].eq(0).sum()))

    return data_animal_id_groups


def time_series_analyis(data):
	'''
	Function to perform time series analysis on provided
	dataset.
	Remove the columns stopped as it has nominal values
	'''

	rm_colm = ['stopped']
	df = data[data.columns.difference(rm_colm)]

	extracted_features = extract_features(
		df, column_id = 'animal_id', column_sort = 'time')

	impute(extracted_features)

	return(extracted_features)


def medoid_computation(data):
	'''
	Calculates the data point (animal_id) closest to
	center/centroid/medoid for a time step
    Uses group by on 'time' attribute

    Input-      Expects a Pandas CSV input parameter containing the dataset
    Returns-    Pandas DataFrame containing computed medoids & centroids
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




def parse_csv(path_to_file):
    '''
    A function to read CSV file into a Pandas DataFrame-
    Expects complete path/relative path to CSV file along with file name
    '''
    try:

        if path_to_file[-3:] == 'csv':
            data = pd.read_csv(path_to_file)
        else:
            data = pd.read_csv(path_to_file + '.csv')

        # change column names all to lower case values
        data.columns = map(str.lower, data.columns)

        # check if all required columns are there in the right format
        if 'time' in data and 'animal_id' in data and 'x' in data and 'y' in data:
            # Check if 'time' attribute is integer-
            if is_numeric_dtype(data['time']):
                data.sort_values('time', ascending=True, inplace=True)
                # Check if 'time' attribute is string-
            elif is_string_dtype(data['time']):
                data['time'] = pd.to_datetime(data['time'])
                data.sort_values('time', ascending=True, inplace=True)

            # Check if 'heading_angle' attribute is given in CSV file-
            if 'heading_angle' in data and np.issubdtype(data['heading_angle'].dtype, np.number):
                print("\n'heading_angle' attribute is found (numeric type) and will be processed\n")
                # do nothing, as 'heading_angle' attribute exists
            else:
                print("\nWARNING: 'heading_angle' attribute is not found in the given CSV data file. Continuing without it!\n")


            return data
    except FileNotFoundError:
        print(
            "Your file below could not be found.\nPath given: {0}\n\n".format(
                path_to_file))
    except EmptyDataError:
        print(
            'Your file is empty, has no header, or misses some required columns.'
        )


def parse_excel(path_to_file):
    '''
    Function to read Excel file into a Pandas DataFrame-
    Expects complete path/relative path to CSV file along with file name

    Expects package 'xlrd' to be installed for this to work!
    '''
    try:

        if path_to_file[-3:] == 'xls' or path_to_file[-4:] == 'xlsx':
            data = pd.read_excel(path_to_file)
        else:
            data = pd.read_excel(path_to_file + '.xlsx')

        # change column names all to lower case values
        data.columns = map(str.lower, data.columns)

        # check if all required columns are there in the right format
        if 'time' in data and 'animal_id' in data and 'x' in data and 'y' in data:
            # Check if 'time' attribute is integer-
            if is_numeric_dtype(data['time']):
                data.sort_values('time', ascending=True, inplace=True)
                # Check if 'time' attribute is string-
            elif is_string_dtype(data['time']):
                data['time'] = pd.to_datetime(data['time'])
                data.sort_values('time', ascending=True, inplace=True)

            # Check if 'heading_angle' attribute is given in CSV file-
            if 'heading_angle' in data and np.issubdtype(data['heading_angle'].dtype, np.number):
                print("\n'heading_angle' attribute is found (numeric type) and will be processed\n")
                # do nothing, as 'heading_angle' attribute exists
            else:
                print("\nWARNING: 'heading_angle' attribute is not found in the given CSV data file. Continuing without it!\n")


            return data

    except FileNotFoundError:
        print(
            "Your file below could not be found.\nPath given: {0}\n\n".format(
                path_to_file))
    except EmptyDataError:
        print(
            'Your file is empty, has no header, or misses some required columns.'
        )




import math
import matplotlib.pyplot as plt
import seaborn as sns


def linear_interpolation(data, threshold):
	'''
	Function to interpolate missing values for 'x' and 'y' attributes
	in dataset.
	'threshold' parameter decides the number of rows till which, data
	should NOT be deleted.

	Input- Accepts Pandas DataFrame
	Returns- Processed Pandas DataFrame
	'''

	# Get indices of missing values for 'x' attribute in a list-
	missing_x_values = list(data[data['x'].isnull()].index)

	# Get indices of missing values for 'y' attribute in a list-
	missing_y_values = list(data[data['y'].isnull()].index)

	print("\nNumber of missing values in 'x' attribute = {0}". \
		format(len(missing_x_values)))
	print("Number of missing values in 'y' attribute = {0}\n". \
		format(len(missing_y_values)))


	# counter for outer loop-
	i = 0

	# counter for inner loop-
	j = 0

	# start & end counters-
	start = end = 0

	# count length of sequence found-
	k = 1

	# list containing indices to be deleted-
	indices_to_delete = []

	# threshold = 10

	while i < (len(missing_x_values) - 1):
		start = end = missing_x_values[i]
		k = 1
		# j = missing_x_values[i]
		j = i

		# print("\ni = {0} & j = {1}".format(i, j))

		while j < (len(missing_x_values) - 1):
			# print("j = ", j)
			if missing_x_values[j] + 1 == missing_x_values[j + 1]:
				k += 1
				j += 1
				# end = j
				end = missing_x_values[j]
			else:
				# i = j + 1
				break

		i = j + 1

		if k >= threshold:
			# Delete rows-
			print("\nDelete sequence from {0} to {1}\n".format(start, end))
			# data = data.drop(data.index[start:end + 1])
			# data.drop(data.index[start: end + 1], inplace = True, axis = 0)
			for x in range(start, end + 1):
				indices_to_delete.append(x)

		elif k > 1:
			# Perform Linear Interpolation-
			print("\nSequence length = {0}. Start = {1} & End = {2}". \
				format(k, start, end))

	# Delete indices-
	data_del = data.drop(data.index[indices_to_delete], axis=0)

	return data_del


def plot_missing_values(data, animal_id):
    """
    Plot the missing values of an animal id against the time

    Input:
    'data' is the Pandas Data Frame containing CSV file
    'animal_id' is the ID of the animal who's missing values have to be plotted
    against the time

    Returns:
    Nothing
    """

    missing_time = data[data['time'].isnull()].index.tolist()
    missing_x = data[data['x'].isnull()].index.tolist()
    missing_y = data[data['y'].isnull()].index.tolist()

    # This visualizes the location(s) of missing values-
    # sns.heatmap(df.isnull(), cbar=False)
    # sns.heatmap(df.isnull())

    # Visualizing the count of missing values for all attributes-
    data.isnull().sum().plot(kind='bar')
    plt.xticks(rotation=20)
    plt.title("Visualizing count of missing values for all attributes")
    plt.show()

    return None


def clean(data):
    '''
    A function to perform data preprocessing
    Expects 'data' as input which is the Pandas DataFrame to be processed
    '''
    # Print the number of missing values per column
    print_missing(data)

    # Drop columns with  missing values for 'time'  and 'animal_id'
    data.dropna(subset=['animal_id', 'time'], inplace=True)
    # Change column type of animal_id and time
    data['animal_id'] = data['animal_id'].astype(np.int64)
    data['time'] = data['time'].astype(np.int64)

    # Print duplicate rows
    print_duplicate(data)
    # Remove the duplicated rows found above
    data.drop_duplicates(subset=['animal_id', 'time'], inplace=True)

    return data


def print_missing(df):
    '''
    Print the missing values for each column

    Input- Accepts Pandas DataFrame
    Return- No return
    '''
    print("\n Number of missing values = {0}\n". \
    	format(df.isnull().sum().sort_values(ascending=False)))

    return None


def print_duplicate(df):
    '''
    Print the duplicate rows
    '''
    dup = df[df.duplicated(['time', 'animal_id'])]
    print(
        "Removed duplicate rows based on the columns 'animal_id' and 'time' column are:",
        dup,
        sep='\n')


def grouping_data(processed_data):
    '''
    A function to group all values for each 'animal_id'
    Input is 'processed_data' which is processed Pandas DataFrame
    Returns a dictionary where- key is animal_id, value in Pandas DataFrame for that 'animal_id'
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
        data_animal_id_groups[animal_id].reset_index(drop=True, inplace=True)

    # Add additional attributes/columns to each groups-
    for aid in data_animal_id_groups.keys():
        data = [0 for x in range(data_animal_id_groups[aid].shape[0])]

        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            distance=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            average_speed=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            average_acceleration=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            direction=data)

    return data_animal_id_groups


def filter_dataframe(data, frm, to):
    """
    A function to filter the dataset, which is the first
    argument to the function using 'frm' and 'to' as the
    second and third arguments.
    Please note that both 'frm' and 'to' are included in
    the returned filtered Pandas Data frame.

    Returns a filtered Pandas Data frame according to 'frm'
    and 'to' arguments
    """

    return data.loc[(data['time'] >= frm) & (data['time'] < to), :]


def replace_parts_animal_movement(data_groups, animal_id, time_array,
                                  replacement_value_x, replacement_value_y):
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

    An example usage-

    data = csv_to_pandas(path)

    data_groups = group_animals(data)

    arr_index = np.array([10, 20, 200, 20000, 40000, 43200])

    replaced_data_groups = replace_parts_animal_movement(data_groups, 811, arr_index, 100, 90)
    """
    data_groups[animal_id].loc[time_array, 'x'] = replacement_value_x
    data_groups[animal_id].loc[time_array, 'y'] = replacement_value_y

    return data_groups


def resample_systematic(data_groups, downsample_size):
    """
    Resample the movement data of each animal - by downsampling at fixed time
    intervals. This is done to reduce the resolution of dataset
    This function does this by systematically choosing samples from each animal

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
    intervals. This is done to reduce the resolution of dataset
    This function does this by randomly choosing samples from each animal

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

    # Random index (numpy.ndarray)-
    ix_random = np.random.randint(0, size, downsample_size)

    # Modified 'data_groups' downsampled Python 3 dictionary-
    data_groups_downsampled = {}

    for aid in data_groups.keys():
        data_groups_downsampled[aid] = data_groups[aid].loc[ix_random, :]

    return data_groups_downsampled


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


def split_trajectories_fuzzy_segmentation(data_groups,
                                          segment=1,
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


def preprocessing_methods(
	linear_interpolation_fn = False, plot_missing_values_fn = False,
	clean_fn = False, print_missing_fn = False, filter_dataframe_fn = False,
	replace_parts_animal_movement_fn = False,
	resample_systematic_fn = False,
	resample_random_fn = False,
	split_trajectories_fn = False,
	split_trajectories_fuzzy_segmentation_fn = False,
	frm = 0, to = 0, data = 0, threshold = 0,
	data_groups = 0, animal_id = 0, time_array = 0,
	replacement_value_x = 0, replacement_value_y = 0,
	downsample_size = 0, segment = 0, fuzzy_segment = 0
	):
	"""
	Function containing all of preprocessing functions as function
	arguments (which by default are False)
	"""

	if linear_interpolation_fn == True:
		return linear_interpolation(data, 5)

	elif plot_missing_values_fn == True:
		plot_missing_values(data, animal_id)
		return None

	elif print_missing_fn == True:
		print_missing(data)
		return None

	elif clean_fn == True:
		clean(data)
		return None

	elif filter_dataframe_fn == True:
		return filter_dataframe(data, frm, to)

	elif replace_parts_animal_movement_fn == True:
		return replace_parts_animal_movement(data_groups,
			animal_id, time_array, replacement_value_x,
			replacement_value_y)

	elif resample_systematic_fn == True:
		return resample_systematic(
			data_groups, downsample_size)
    
	elif resample_random_fn == True:
		return resample_random(
			data_groups, downsample_size)
		

	elif split_trajectories_fn == True:
		split_trajectories(data_groups, segment=1)
		return None

	elif split_trajectories_fuzzy_segmentation_fn == True:
		split_trajectories_fuzzy_segmentation(
			data_groups, segment=1, fuzzy_segment=2)
		return None


def plot_x_and_y(data, frm, to):
	"""
	Function to plot the features 'x' and 'y'
	for a given Pandas DataFrame 'data'

	Input:
	data 	- Pandas DataFrame (should be sorted by 'time' attribute)
	frm 	- starting from time step
	to 		- ending to time step

	Returns:
	Nothing to return, plots for given parameters
	"""
	# Specify the from and to time steps-
	#frm = 10; to = 15


	plt.scatter(x = 'x', y = 'y', 
		data = data.loc[(data['time'] >= frm) & (data['time'] <= to), :])
	plt.title("Plotting for 'x' and 'y' attributes")
	plt.xlabel("'x' coordinate")
	plt.ylabel("'y' coordinate")
	plt.grid()

	plt.show()

	return None



"""
# -*- coding: utf-8 -*-
#__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from pkg_resources import get_distribution, DistributionNotFound

try:
	# Change here if project is renamed and does not equal the package name
	dist_name = __name__
	__version__ = get_distribution(dist_name).version
except DistributionNotFound:
	__version__ = 'unknown'
finally:
	del get_distribution, DistributionNotFound

"""