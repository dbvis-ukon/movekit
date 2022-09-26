import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import utm
from datetime import datetime
from .utils import presence_3d
from .feature_extraction import grouping_data, regrouping_data


def from_dataframe(data, dictionary):
    """
    Reformat an existing DataFrame to make it compatible with movekit
    :param data: pandas DataFrame. The data to be reformatted
    :param dictionary: Key-value pairs of column names. Keys store the old column names. The respective new column names
    are stored as their values. Values that need to be defined include 'time', 'animal_id', 'x' and 'y'
    :return: pandas DataFrame
    """

    # perform a check
    mandatory = ['time', 'animal_id', 'x', 'y']
    passed = all(elem in dictionary.values() for elem in mandatory)
    if passed:
        return data.rename(mapper=dictionary, axis=1)
    else:
        raise ValueError('Must contain the column names "time", "animal_id", "x" and "y"')


def interpolate(data,
                limit=1,
                limit_direction="forward",
                inplace=False,
                method="linear",
                order=1,
                date_format=False):
    """
    Interpolate over missing values in pandas Dataframe of movement records.
    Interpolation methods consist of "linear", "polynomial, "time", "index", "pad".
    (see  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html)
    :param data: Pandas DataFrame of movement records
    :param limit: Maximum number of consecutive NANs to fill
    :param limit_direction: If limit is specified, consecutive NaNs will be filled in this direction.
    :param inplace: Update the data in place if possible.
    :param method: Interpolation technique to use. Default is "linear".
    :param order: To be used in case of polynomial or spline interpolation.
    :param date_format: Boolean to define whether time is some kind of date format. In this case column type has to be converted before calling interpolate.
    :return: Interpolated DataFrame.
    """
    # converting time column if needed
    if date_format:
        if 'time' in data.columns:
            timestamp_column = data['time'].apply(lambda x: x.timestamp())
            time_difference = timestamp_column.apply(lambda x: datetime.fromtimestamp(x))[data['time'].first_valid_index()]\
            - data['time'][data['time'].first_valid_index()]
            data['time'] = data['time'].apply(lambda x: x.timestamp())
        else:
            warnings.warn('Please rename the time column to "time".')


    # Interpolating record data
    if method != "polynomial" and method != "spline":
        interp = data.interpolate(limit=limit,
                                  limit_direction=limit_direction,
                                  inplace=inplace,
                                  method=method)
    else:
        interp = data.interpolate(limit=limit,
                                  limit_direction=limit_direction,
                                  inplace=inplace,
                                  method=method,
                                  order=order)
    # convert time column back to date
    if date_format:
        interp['time'] = interp['time'].apply(lambda x: datetime.fromtimestamp(x)) - time_difference

    return interp


# Function only plots missings for all animals, therefore dead parameter
def plot_missing_values(data):
    """
    Plot the missing values of an animal-ID against time.
    :param data: Pandas DataFrame containing records of movement.
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
               method="linear",
               order=1,
               date_format = False):
    """
    Function to perform data preprocessing.
    Print the number of missing values per column; Drop columns with missing values for 'time' and 'animal_id';
    Remove the duplicated rows found.
    :param data: DataFrame to perform preprocessing on
    :param dropna: Optional parameter to drop columns with  missing values for 'time' and 'animal_id'
    :param interpolation: Optional parameter to perform interpolation
    :param limit: Maximum number of consecutive NANs to fill
    :param limit_direction: If limit is specified, consecutive NaNs will be filled in this direction.
    :param inplace: Update the  data in place if possible.
    :param method: Interpolation technique to use. Default is "linear".
    :param order: To be used in case of polynomial or spline interpolation.
    :param date_format: Boolean to define whether time is some kind of date format. Important for interpolation.
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
                           method=method,
                           order=order,
                           date_format=date_format)

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
        "Duplicate rows based on the columns 'animal_id' and 'time' column are:",
        dup,
        sep='\n')


def filter_dataframe(data, frm, to):
    """
    Extract records of assigned time frame from preprocessed movement record data.
    :param data: Pandas DataFrame, containing preprocessed movement record data.
    :param frm: Int, defining starting point from where to extract records.Note that if time is stored as a date (if input data has time not stored as numeric type it is automatically converted to datetime) parameter has to be set using an datetime format: mkit.filter_dataframe(data, "2008-01-01", "2010-10-01")
    :param to: Int, defining end point up to where to extract records.
    :return: Pandas DataFrame, filtered by records matching the defined frame in 'from'-'to'.
    """
    return data.loc[(data['time'] >= frm) & (data['time'] <= to), :]


def replace_parts_animal_movement(data_groups, animal_id, time_array,
                                  replacement_value_x, replacement_value_y, replacement_value_z=None):
    """
    Replace subsets (segments) of animal movement based on some indices e.g. time.
    This function can be used to remove outliers.

    Example usage:
        data_groups = grouping_data(data)
        arr_index = np.array([10, 20, 200, 20000, 40000, 43200])
        replaced_data_groups = replace_parts_animal_movement(data_groups, 811, arr_index, 100, 90)

    :param data_groups: DataFrame containing the movement records.
    :param animal_id: Int defining 'animal_id' whose movements have to be replaced.
    :param time_array: Array defining time indices whose movements have to replaced (array of integers if time has integer format, array of strings with datetime if time is datetime format)
    :param replacement_value_x: Int value that will replace all 'x' attribute values in 'time_array'.
    :param replacement_value_y: Int value that will replace all 'y' attribute values in 'time_array'.
    :param replacement_value_z: Int value that will replace all 'z' attribute values in 'time_array'. (optional)
    :return: Dictionary with replaced subsets.
    """
    # Grouping DataFrame
    data_groups = grouping_data(data_groups, preprocessedMethod=True)

    data_groups[animal_id].loc[data_groups[animal_id]["time"].isin(time_array), 'x'] = replacement_value_x
    data_groups[animal_id].loc[data_groups[animal_id]["time"].isin(time_array), 'y'] = replacement_value_y
    if presence_3d(data_groups[animal_id]):
        data_groups[animal_id].loc[data_groups[animal_id]["time"].isin(time_array), 'z'] = replacement_value_z

    return regrouping_data(data_groups)


def resample_systematic(data_groups, downsample_size):
    """
    Resample the movement data of each animal - by downsampling at fixed time intervals.
    This is done to reduce the resolution of the dataset. This function does this by systematically choosing
    samples from each animal.
    :param data_groups: DataFrame containing the movement records.
    :param downsample_size: Int sample size to which each animal has to be reduced by downsampling.
    :return: DataFrame, modified from original size 'data_groups' to 'downsample_size'.
    """
    # group the dataFrame
    data_groups = grouping_data(data_groups, preprocessedMethod=True)

    # Get first key-
    first = list(data_groups.keys())[0]

    # size of each animal's group-
    size = data_groups[first].shape[0]

    step_size = math.floor(size / downsample_size)

    l = list(range(size))
    arr_index = l[0:(step_size * downsample_size):step_size]

    # Convert list to numpy array-
    arr_index = np.asarray(arr_index)

    # Modified 'data_groups' downsampled Python 3 dictionary-
    data_groups_downsampled = {}

    for aid in data_groups.keys():
        data_groups_downsampled[aid] = data_groups[aid].loc[arr_index, :]

    data_groups_downsampled = regrouping_data(data_groups_downsampled)

    return data_groups_downsampled


def resample_random(data_groups, downsample_size):
    """
    Resample the movement data of each animal - by downsampling at random time intervals.
    This is done to reduce resolution of the dataset. This function does this by randomly choosing
    samples from each animal.
    :param data_groups: DataFrame containing the movement records.
    :param downsample_size: Int sample size to which each animal has to be reduced by downsampling.
    :return: DataFrame, modified from original size 'data_groups' to 'downsample_size'.
    """
    # group the dataFrame
    data_groups = grouping_data(data_groups, preprocessedMethod=True)

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

    data_groups_downsampled = regrouping_data(data_groups_downsampled)

    return data_groups_downsampled


def split_trajectories(data_groups, segment, fuzzy_segment=0, csv=False):
    """
    Split trajectory of a single animal into several segments based on specific criterion.

    Example usage:
        data_groups = group_animals(data)
        split_trajectories_fuzzy_segmentation(data_groups, segment = 5, fuzzy_segment = 5)

    :param data_groups: DataFrame with movement records.
    :param segment: Int, defining point where the animals are split into several Pandas Data Frames.
    :param fuzzy_segment: Int, defining interval which will overlap on either side of the segments.
    :param csv: Boolean, defining if each interval shall be exported locally as singular csv
    :return: Dictionary with the created DataFrames for each animal.
    """
    # Group the DataFrame
    data_groups = grouping_data(data_groups, preprocessedMethod=True)

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


def convert_measueres(preprocessed_data, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1):
    """
    Create a linear scale with input parameters for x,y for transformation of position data.
    :param preprocessed_data: Pandas DataFrame only with x and y position data
    :param x_min: int minimum for x - default: 0.
    :param x_max: int maximum for x - default: 1.
    :param y_min: int minimum for y - default: 0.
    :param y_max: int maximum for y - default: 1.
    :param z_min: int minimum for z - default: 0.
    :param z_max: int maximum for z - default: 1.
    :return: Pandas DataFrame with linearly transformed position data.
    """
    # Preventing features input along position data
    if [*preprocessed_data.columns] != ['time', 'animal_id', 'x', 'y'] and [*preprocessed_data.columns] != ['time', 'animal_id', 'x', 'y', 'z']:
        print(
            "\nError! Conversion only allowed for dataframes with colnames ['time', 'animal_id', 'x', "
            "'y'] or ['time', 'animal_id', 'x', 'y', 'z']. \n")
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

    if presence_3d(preprocessed_data):
        preprocessed_data.loc[:, "z"] = np.interp(
            preprocessed_data.loc[:, "z"], (preprocessed_data.loc[:, "z"].min(),
                                            preprocessed_data.loc[:, "z"].max()),
            (z_min, z_max))
    return preprocessed_data


def normalize(data):
    """
    Normalizes values for the 'x' and 'y' column
    :param data: DataFrame to perform preprocessing on
    :return: normalized DataFrame
    """
    data['x'] = (data['x'] - data['x'].min()) / (data['x'].max() - data['x'].min())
    data['y'] = (data['y'] - data['y'].min()) / (data['y'].max() - data['y'].min())
    if presence_3d(data):
        data['z'] = (data['z'] - data['z'].min()) / (data['z'].max() - data['z'].min())
    return data


def delete_mover(data, animal_id):
    """
    Delete a particular mover from the DataFrame
    :param data: DataFrame
    :param animal_id: int. The animal_id as found in the column animal_id
    :return: DataFrame
    """
    return data.drop(data[data['animal_id'] == animal_id].index)


def convert_latlon(data, latitude='latitude', longitude='longitude', replace=True):
    """
    Project data from GPS coordinates (latitude and longitude) to the cartesian coordinate system
    :param data: DataFrame with GPS coordinates
    :param latitude: str. Name of the column where latitude is stored
    :param longitude: str. Name of the column where longitude is stored
    :param replace: bool. Flag whether the xy columns should replace the latlon columns
    :return: DataFrame after the transformation where latitude is projected into y and longitude is projected into x
    """

    # get utm zone to check if all points are in same utm zone
    utm_coord = utm.from_latlon(data[latitude].iloc[0], data[longitude].iloc[0])
    zone = utm_coord[2]

    data['x'] = np.nan
    data['y'] = np.nan

    for i, row in data.iterrows():
        # get the xy coordinates
        utm_coord = utm.from_latlon(row[latitude], row[
            longitude])  # utm converts a (latitude, longitude) tuple into the form (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER
        x = utm_coord[0]
        y = utm_coord[1]
        # add to dataFrame
        data.at[i, 'x'] = x
        data.at[i, 'y'] = y

        # issue warning if unseen zone
        if utm_coord[2] != zone:
            warnings.warn("Input data spans multiple UTM zones. Projection into plane will likely be inaccurate.")

    if replace:
        data.drop(latitude, axis=1, inplace=True)
        data.drop(longitude, axis=1, inplace=True)

    return data
