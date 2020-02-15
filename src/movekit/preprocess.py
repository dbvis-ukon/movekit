import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


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

    # Visualizing the count of missing values for all attributes-
    data.isnull().sum().plot(kind='bar')
    plt.xticks(rotation=20)
    plt.title("Visualizing count of missing values for all attributes")
    plt.show()

    return None


def preprocess(data):
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

    return groups


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
    return groups
