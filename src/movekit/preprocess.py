import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Function does not interpolate
def linear_interpolation(data, threshold):
    """
    Interpolate missing values for 'x' and 'y' attributes in dataset.

    :param data: Pandas DataFrame with movement records.
    :param threshold: Integer to define the number of rows till which data should NOT be deleted.
    :return: Processed Pandas DataFrame.
    """
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

# DEAD VARIABLES?
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

# Function only plots missings for all animals, therefore dead parameter
def plot_missing_values(data, animal_id):
    """
    Plot the missing values of an animal-ID against time.

    :param data: Pandas DataFrame containing records of movement.
    :param animal_id: ID of the animal whose missing values will be plotted against time.
    :return: None.
    """

# DEAD VARIABLES?
    missing_time = data[data['time'].isnull()].index.tolist()
    missing_x = data[data['x'].isnull()].index.tolist()
    missing_y = data[data['y'].isnull()].index.tolist()

    # Visualizing the count of missing values for all attributes-
    data.isnull().sum().plot(kind='bar')
    plt.xticks(rotation=20)
    plt.title("Visualizing count of missing values for all attributes")
    plt.show()

    return None

# Is this complete? If this is a combined function to preprocess, we might want to include interpolation
def preprocess(data):
    """
    A function to perform data preprocessing.

    Print the number of missing values per column; Drop columns with  missing values for 'time' and 'animal_id';
    Remove the duplicated rows found.

    :param data: Pandas DataFrame to be processed.
    :return: DataFrame processed.
    """
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
    """
    Print the missing values for each column.

    :param df: Pandas DataFrame of movement records.
    :return: None.
    """
    print("Total number of missing values = ", df.isnull().sum().sum())
    print(format(df.isnull().sum().sort_values(ascending=False)))

    return None


def print_duplicate(df):
    """
    Print rows, which are duplicates.

    :param df: Pandas DataFrame of movement records.
    :return: None.
    """
    dup = df[df.duplicated(['time', 'animal_id'])]
    print(
        "Removed duplicate rows based on the columns 'animal_id' and 'time' column are:",
        dup,
        sep='\n')

def filter_dataframe(data, frm, to):
    """
    Extract records of assigned time frame from preprocessed movement record data.

    :param data: Pandas DataFrame, containing preprocessed movement record data.
    :param frm: Int, defining starting point from where to extract records.
    :param to: Int, defining end point up to where to extract records.
    :return: Pandas DataFrame, filtered by records matching the defined frame in 'from'-'to'.
    """
    return data.loc[(data['time'] >= frm) & (data['time'] <= to), :]


def replace_parts_animal_movement(data_groups, animal_id, time_array,
                                  replacement_value_x, replacement_value_y):
    """
    Replace subsets (segments) of animal movement based on some indices e.g. time.
    This function can be used to remove outliers.

    Example usage:
        data_groups = grouping_data(data)
        arr_index = np.array([10, 20, 200, 20000, 40000, 43200])
        replaced_data_groups = replace_parts_animal_movement(data_groups, 811, arr_index, 100, 90)

    :param data_groups: Dictionary with key 'animal_id'and value with records for 'animal_id'.
    :param animal_id: Int defining 'animal_id' whose movements have to be replaced.
    :param time_array: Array defining time indices whose movements have to replaced
    :param replacement_value_x: Int value that will replace all 'x' attribute values in 'time_array'.
    :param replacement_value_y: Int value that will replace all 'y' attribute values in 'time_array'.
    :return: Dictionary with replaced subsets.
    """
    data_groups[animal_id].loc[time_array, 'x'] = replacement_value_x
    data_groups[animal_id].loc[time_array, 'y'] = replacement_value_y

    return data_groups


def resample_systematic(data_groups, downsample_size):
    """
    Resample the movement data of each animal - by downsampling at fixed time intervals.

    This is done to reduce the resolution of the dataset. This function does this by systematically choosing
    samples from each animal.

    :param data_groups: Dictionary with key: 'animal_id' and value with record data to that 'animal_id'.
    :param downsample_size: Int sample size to which each animal has to be reduced by downsampling.
    :return: Dictionary, modified from original size 'data_groups' to 'downsample_size'.
    """
    # Get first key-
    first = list(data_groups.keys())[0]

    # size of each animal's group-
    size = data_groups[first].shape[0]

    step_size = math.floor(size / downsample_size)

# DEAD LIST?
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
    Resample the movement data of each animal - by downsampling at random time intervals.

    This is done to reduce resolution of the dataset. This function does this by randomly choosing
    samples from each animal

    :param data_groups: Dictionary with key 'animal_id' and value record data of 'animal_id'.
    :param downsample_size: Int sample size to which each animal has to be reduced by downsampling.
    :return: Dictionary, modified from original size 'data_groups' to 'downsample_size'.
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


def split_trajectories(data_groups, segment, fuzzy_segment = 0, csv = False):
    """
    Split trajectory of a single animal into several segments based on specific criterion.

    Example usage:
        data_groups = group_animals(data)
        split_trajectories_fuzzy_segmentation(data_groups, segment = 5, fuzzy_segment = 5)

    :param data_groups: Dictionary with key 'animal_id' and value record data of 'animal_id'.
    :param segment: Int, defining point where the animals are split into several Pandas Data Frames.
    :param fuzzy_segment: Int, defining interval which will overlap on either side of the segments.
    :param csv: Boolean, defining if each interval shall be exported locally as singular csv
    :return: None. All segmented Pandas Data Frames are saved to HDD.
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

    if csv == True:
        for k in groups.keys():
            groups[k].to_csv(k + '.csv', index=False)

    return groups
'''

'''


