import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def interpolate(data,
                limit=1,
                limit_direction="forward",
                inplace=False,
                method="linear"):
    """
    Interpolate over missing values in pandas Dataframe of movement records.
    Interpolation methods consist of "linear", "polynomial, "time", "index", "pad".
    (see  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html)

    :param data: Pandas DataFrame of movement records
    :param limit: Maximum number of consecutive NANs to fill
    :param limit_direction: If limit is specified, consecutive NaNs will be filled in this direction.
    :param method: Interpolation technique to use. Default is "linear".
    :param order: To be used in case of polynomial interpolation.
    :return: Interpolated DataFrame.
    """
    # Interpolating record data
    interp = data.interpolate(limit=limit,
                              limit_direction=limit_direction,
                              inplace=inplace,
                              method=method)
    return interp


# Function only plots missings for all animals, therefore dead parameter
def plot_missing_values(data):
    """
    Plot the missing values of an animal-ID against time.

    :param data: Pandas DataFrame containing records of movement.
    :param animal_id: ID of the animal whose missing values will be plotted against time.
    :return: None.
    """
    # Visualizing the count of missing values for all attributes-
    data.isnull().sum().plot(kind='bar')
    plt.xticks(rotation=20)
    plt.title("Visualizing count of missing values for all attributes")
    plt.show()

    return None


def preprocess(data,
               dropna=True,
               interpolation=False,
               limit=1,
               limit_direction="forward",
               inplace=False,
               method="linear"):
    """
    Function to perform data preprocessing.

    Print the number of missing values per column; Drop columns with  missing values for 'time' and 'animal_id';
    Remove the duplicated rows found.
    :param data: DataFrame to perform preprocessing on
    :param dropna: Optional parameter to drop columns with  missing values for 'time' and 'animal_id'
    :param interpolate: Optional parameter to perform linear interpolation
    :param limit: Maximum number of consecutive NANs to fill
    :param limit_direction: If limit is specified, consecutive NaNs will be filled in this direction.
    :param method: Interpolation technique to use. Default is "linear".
    :param order: To be used in case of polynomial interpolation.
    :return: Preprocessed DataFrame.
    """
    # Print the number of missing values per column
    print_missing(data)

    # Interpolate data with missings
    if interpolation:
        data = interpolate(data,
                           limit=limit,
                           limit_direction=limit_direction,
                           inplace=inplace,
                           method=method)

    # Drop columns with  missing values for 'time'  and 'animal_id'

    if dropna:
        data.dropna(subset=['animal_id', 'time'], inplace=True)

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


def split_trajectories(data_groups, segment, fuzzy_segment=0, csv=False):
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


def convert_measueres(preprocessed_data, x_min=0, x_max=1, y_min=0, y_max=1):
    """
    Create a linear scale with input parameters for x,y for transformation of position data.
    :param preprocessed_data: Pandas DataFrame only with x and y position data
    :param x_min: int minimum for x - default: 0.
    :param x_max: int maximum for x - default: 1.
    :param y_min: int minimum for y - default: 0.
    :param y_max: int maximum for y - default: 1.
    :return: Pandas DataFrame with linearly transformed position data.
    """
    # Preventing features input along position data
    if [*preprocessed_data.columns] != ['time', 'animal_id', 'x', 'y']:
        print(
            "\nError! Conversion only allowed for dataframes with colnames ['time', 'animal_id', 'x', "
            "'y']. \n")
        return None

    # Linear Transformation of position dimensions
    preprocessed_data.loc[:, "x"] = np.interp(
        preprocessed_data.loc[:, "x"], (preprocessed_data.loc[:, "x"].min(),
                                        preprocessed_data.loc[:, "x"].max()),
        (x_min, x_max))
    preprocessed_data.loc[:, "y"] = np.interp(
        preprocessed_data.loc[:, "y"], (preprocessed_data.loc[:, "y"].min(),
                                        preprocessed_data.loc[:, "y"].max()),
        (y_min, y_max))
    return preprocessed_data
