import numpy as np
import pandas as pd
import warnings
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import tsfresh
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from geoalchemy2 import functions, elements
from .utils import presence_3d, angle
from tqdm import tqdm
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import directed_hausdorff
import multiprocessing
from functools import partial
import re
from pandas.api.types import is_numeric_dtype



def grouping_data(processed_data, pick_vars=None, preprocessedMethod=False):
    """
    Function to group data records by 'animal_id'. Adds additional attributes/columns, if features aren't extracted yet.
    :param processed_data: pd.DataFrame with all preprocessed records.
    :param preprocessedMethod: Boolean whether calling method is from preprocessing to check whether columns for features are added.
    :return: dictionary with 'animal_id' as key and all records as value.
    """

    # A dictionary object to hold all groups obtained using group by-
    # Apply grouping using 'animal_id' attribute-
    data_animal_id = processed_data.groupby('animal_id')

    # A dictionary object to hold all groups obtained using group by-
    data_animal_id_groups = {}

    # Get each animal_id's data from grouping performed-
    # for animal_id in data_animal_id.groups.keys():
    #    data_animal_id_groups[animal_id] = data_animal_id.get_group(animal_id)

    for animal_id, df in data_animal_id:
        data_animal_id_groups[animal_id] = df

    # To reset index for each group-
    for animal_id in data_animal_id_groups.keys():
        data_animal_id_groups[animal_id].reset_index(drop=True, inplace=True)
    if list(processed_data.columns.values) == list(['time', 'animal_id', 'x', 'y']) and not preprocessedMethod:
        # Add additional attributes/columns to each groups-
        for aid in data_animal_id_groups.keys():
            data = [None for x in range(data_animal_id_groups[aid].shape[0])]
            data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
                distance=data)
            data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
                average_speed=data)
            data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
                average_acceleration=data)
            # data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            #     positive_acceleration=data)
            data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
                direction=data)
            data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
                stopped=data)
    if pick_vars != None:
        for aid in data_animal_id_groups.keys():
            data_animal_id_groups[aid] = data_animal_id_groups[
                                             aid].loc[:, pick_vars]
    return data_animal_id_groups


def timewise_dict(data):
    """
    Group records by timestep.
    :param data: pd.DataFrame with all preprocessed records.
    :return: dictionary with 'time' as key and all animal records as value.
    """
    # Input grouped into time
    data_time = data.groupby('time')

    # Pour into dictionary to hold data by time
    dict_time = {}

    for time in data_time.groups.keys():
        dict_time[time] = data_time.get_group(time)
        dict_time[time].reset_index(drop=True, inplace=True)

    return dict_time


def regrouping_data(data_animal_id_groups):
    """
    Concatenate all Pandas DataFrames in grouped dictionary into one.
    :param data_animal_id_groups: dictionary ordered by 'animal_id'.
    :return: Pandas DataFrame containing all records of all andimal_ids.
    """
    # Concatenate all single records into one
    result = pd.concat(data_animal_id_groups[aid]
                       for aid in data_animal_id_groups.keys())

    result.sort_values(['time', 'animal_id'], ascending=True, inplace=True)
    # Reset index-
    result.reset_index(drop=True, inplace=True)
    return result


def compute_direction_angle(data, param_x='x', param_y='y', colname='direction_angle'):
    """
    Computes the angle of rotation of an animal between two timesteps. Only possible if coordinates are 2D only.
    :param data: dataframe containing the movement records
    :param param_x: column name of the x coordinate
    :param param_y: column name of the y coordinate
    :param colname: the name to appear in the new DataFrame for the direction angle computed.
    :return: dataframe containing computed 'direction_angle' as angle from 0-360 degrees (x-axis to the right is 0 degrees)
    """

    data_animal_id_groups = grouping_data(data)

    # take the first dataframe to check the 3d presence
    if presence_3d(data_animal_id_groups[next(iter(data_animal_id_groups))]):
        is_3d = True
        warnings.warn(
            'The direction angle can only be calculated for two-dimensional coordinate data. The dataframe given to the function has more than two dimensions.')
    else:
        # iterate over movers
        for aid in data_animal_id_groups.keys():
            data_animal_id_groups[aid]['y_change'] = data_animal_id_groups[aid][param_y] - data_animal_id_groups[aid][
                param_y].shift(periods=1)  # change of y and x coordinate to create direction vector
            data_animal_id_groups[aid]['x_change'] = data_animal_id_groups[aid][param_x] - data_animal_id_groups[aid][
                param_x].shift(periods=1)
            data_animal_id_groups[aid][colname] = data_animal_id_groups[aid]['y_change'] / data_animal_id_groups[aid][
                'x_change']  # formula: tan^(-1) (y_change / x_change) = angle of direction change
            data_animal_id_groups[aid][colname] = data_animal_id_groups[aid][colname].apply(
                lambda x: math.degrees(math.atan(x)))  # convert angle to degrees
            data_animal_id_groups[aid].loc[(data_animal_id_groups[aid]['y_change'] >= 0) & (
                        data_animal_id_groups[aid]['x_change'] < 0), colname] = 180 + data_animal_id_groups[aid][
                colname]  # adjust to correct angle
            data_animal_id_groups[aid].loc[(data_animal_id_groups[aid]['y_change'] < 0) & (
                        data_animal_id_groups[aid]['x_change'] >= 0), colname] = 360 + data_animal_id_groups[aid][
                colname]
            data_animal_id_groups[aid].loc[(data_animal_id_groups[aid]['y_change'] < 0) & (
                        data_animal_id_groups[aid]['x_change'] < 0), colname] = 180 + data_animal_id_groups[aid][
                colname]
            data_animal_id_groups[aid].loc[0, [colname]] = 0  # direction for first timestamp is 0
            data_animal_id_groups[aid].drop(['y_change', 'x_change'], inplace=True, axis=1)
        data = regrouping_data(data_animal_id_groups)
    return data


def compute_turning_angle(data, colname='turning_angle', direction_angle_name='direction_angle'):
    """
    Computes the turning angle for a mover between two timesteps as the difference of its direction angle. Only possible for 2D data.
    :param data: dataframe containing the movement records.
    :param colname: the name of the new column to be added.
    :param direction_angle_name: the name of the column containg the direction angle for each movement record.
    :return: dataframe containing an additional column with the difference in degrees between current and previous timestamp for each record.
    Note that difference can not be higher than +-180 degrees.
    """

    # when differences exceed |180| convert values so that they stay in the domain -180 +180
    boundary = lambda data: 360 - data if data > 180 else -360 - data if data < -180 else data
    vboundary = np.vectorize(boundary)

    # check dataframe if data contains direction_angle column
    if direction_angle_name not in data.columns:
        warnings.warn('As it is needed to calculate the turning angle at first the direction angle has to be computed.')
        data = compute_direction_angle(data)

    data_animal_id_groups = grouping_data(data)
    # iterate over movers
    for aid in data_animal_id_groups.keys():
        turning_angles = data_animal_id_groups[aid][direction_angle_name] - data_animal_id_groups[aid][
            direction_angle_name].shift(
            periods=1)  # calculate difference in direction angle between current and previous timestamp
        turning_angles = vboundary(turning_angles)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            inp=turning_angles)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].rename(
            columns={'inp': colname})
    data = regrouping_data(data_animal_id_groups)
    return data


def compute_direction(data_animal_id_groups, pbar, param_x='x', param_y='y', param_z='z', colname='direction'):
    """
    Computes the movement vector for each timestamp by checking the difference of the coordinates to the previous timestamp.
    :param data_animal_id_groups: dictionary containing the data frames for each animal
    :param pbar: percentage bar filled with 10% already
    :param param_x: column name of the x coordinate
    :param param_y: column name of the y coordinate
    :param param_z: column name of the z coordinate
    :param colname: the name to appear in the new DataFrame for the calculated direction
    :return: dictionary containing computed 'direction' attribute
    """

    percent_update = 90 / len(data_animal_id_groups.keys())  # how much the pb is updated after each animal

    # take the first dataframe to check the 3d presence
    if presence_3d(data_animal_id_groups[next(iter(data_animal_id_groups))]):
        is_3d = True
    else:
        is_3d = False

    # iterate over movers
    for aid in data_animal_id_groups.keys():
        if is_3d:
            data_animal_id_groups[aid]['x_change'] = round(
                data_animal_id_groups[aid][param_x] - data_animal_id_groups[aid][param_x].shift(periods=1), 4)
            data_animal_id_groups[aid]['y_change'] = round(
                data_animal_id_groups[aid][param_y] - data_animal_id_groups[aid][param_y].shift(periods=1), 4)
            data_animal_id_groups[aid]['z_change'] = round(
                data_animal_id_groups[aid][param_z] - data_animal_id_groups[aid][param_z].shift(periods=1), 4)
            data_animal_id_groups[aid].loc[0, 'x_change'], data_animal_id_groups[aid].loc[0, 'y_change'], \
            data_animal_id_groups[aid].loc[0, 'z_change'] = 0, 0, 0
            data_animal_id_groups[aid][colname] = data_animal_id_groups[aid].apply(
                lambda row: (row.x_change, row.y_change, row.z_change), axis=1)
            data_animal_id_groups[aid].drop(['x_change', 'y_change', 'z_change'], inplace=True, axis=1)

        else:
            data_animal_id_groups[aid]['x_change'] = round(
                data_animal_id_groups[aid][param_x] - data_animal_id_groups[aid][param_x].shift(periods=1), 4)
            data_animal_id_groups[aid]['y_change'] = round(
                data_animal_id_groups[aid][param_y] - data_animal_id_groups[aid][param_y].shift(periods=1), 4)
            data_animal_id_groups[aid].loc[0, 'x_change'], data_animal_id_groups[aid].loc[0, 'y_change'] = 0, 0
            data_animal_id_groups[aid][colname] = data_animal_id_groups[aid].apply(
                lambda row: (row.x_change, row.y_change), axis=1)
            data_animal_id_groups[aid].drop(['x_change', 'y_change'], inplace=True, axis=1)

        pbar.update(percent_update)

    return data_animal_id_groups


def compute_turning(data_animal_id_groups, param_direction="direction", colname="turning"):
    """
    Computes the turning for a mover between two timesteps as the cosine similarity between its direction vectors.
    :param data_animal_id_groups: dictionary ordered by 'animal_id'.
    :param param_direction: Column name to be recognized as direction. Default "direction".
    :param colname: the name of the new column added  which contains the computed cosine similarity.
    :return: data_animal_id_groups
    """
    for aid in data_animal_id_groups.keys():
        cos_similarities = [cosine_similarity(np.array([data_animal_id_groups[aid][param_direction][i]]),
                                              np.array([data_animal_id_groups[aid][param_direction][i + 1]]))[0][0] for
                            i in range(0, len(data_animal_id_groups[aid][
                                                  param_direction]) - 1)]  # cosine similarity for direction vectors of two following timestamps
        cos_similarities.insert(0,
                                0)  # first entry has no previous entry -> cosine similarity can not be calculated and value is set to 0
        data_animal_id_groups[aid][colname] = cos_similarities
    return data_animal_id_groups


def compute_distance(data_animal_id_groups, param_x="x", param_y="y", param_z="z"):
    """
    Calculate metric distance of animals in between two timesteps.
    :param data_animal_id_groups: dictionary ordered by 'animal_id'.
    :param param_x: Column name to be recognized as x. Default "x".
    :param param_y: Column name to be recognized as y. Default "y".
    :param param_z: Column name to be recognized as z. Default "z".
    :return: dictionary containing computed 'distance' attribute.
    """
    # take the first dataframe to check the 3d presence
    if presence_3d(data_animal_id_groups[next(iter(data_animal_id_groups))]):
        for aid in data_animal_id_groups.keys():
            p1 = data_animal_id_groups[aid].loc[:, [param_x, param_y, param_z]]
            p2 = data_animal_id_groups[aid].loc[:, [param_x, param_y, param_z]].shift(
                periods=1)
            p2.iloc[0, :] = [0.0, 0.0, 0.0]

            data_animal_id_groups[aid]['distance'] = ((p1 -
                                                       p2) ** 2).sum(axis=1) ** 0.5
    # 2D dataset
    else:
        for aid in data_animal_id_groups.keys():
            p1 = data_animal_id_groups[aid].loc[:, [param_x, param_y]]
            p2 = data_animal_id_groups[aid].loc[:, [param_x, param_y]].shift(
                periods=1)
            p2.iloc[0, :] = [0.0, 0.0]

            data_animal_id_groups[aid]['distance'] = ((p1 -
                                                       p2) ** 2).sum(axis=1) ** 0.5

    # Reset first entry for each 'animal_id' to zero-
    for aid in data_animal_id_groups.keys():
        data_animal_id_groups[aid].loc[0, 'distance'] = 0.0

    return data_animal_id_groups


def compute_average_speed(data_animal_id_groups, fps):
    """
    Compute average speed of mover based on fps parameter. The formula used for calculating average speed is: (Total Distance traveled) / (Total time taken).
    Size of traveling window is determined by fps parameter:
    By choosing f.e. fps=4 at timestamp 5: (distance covered from timestamp 3 to timestamp 7) / 4.
    By choosing f.e. fps=3 at timestamp 5: (distance covered from timestamp 3.5 to timestamp 6.5) / 3. (in this case use of interpolation if timestamps 3.5 and 6.5 do not exist.)
    :param data_animal_id_groups: dictionary with 'animal_id' as keys.
    :param fps: integer to define size of window for integer-formatted time or string to define size of window for datetime-formatted time (For possible units refer to:https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.)
    :return: dictionary, including measure for 'average_speed'.
    """
    if (isinstance(fps, int) or isinstance(fps, float)):
        fps = str(fps) + 'us'  # as integer time is converted to time for flexible windows so is fps
    for aid in data_animal_id_groups.keys():
        # set index as time to have flexible sized windows
        if is_numeric_dtype(data_animal_id_groups[aid]['time']):
            data_animal_id_groups[aid].index = pd.to_datetime(data_animal_id_groups[aid]['time'], unit='us')
        else:
            data_animal_id_groups[aid].set_index('time', drop=False, inplace=True)

        # add distances of window (will later be divided by time)
        data_animal_id_groups[aid]['average_speed'] = data_animal_id_groups[
            aid]['distance'].rolling(min_periods=1, window=fps,
                                     center=True, closed='both').sum().fillna(0)

        timedelta = pd.to_timedelta(fps)  # size time window
        timedelta_left = timedelta / 2  # size window left of observation
        timedelta_right = timedelta / 2  # size window right of observation

        # adjust first term of sum (as not entire distance of first entry in window was moved in time window)
        ind = data_animal_id_groups[aid].index.searchsorted(data_animal_id_groups[aid].index - timedelta_left, side='left')  # find index of first element in window
        out_of_range = (ind <= 0)
        ind[out_of_range] = 1  # otherwise key error
        lag = data_animal_id_groups[aid]['distance'].values[ind]
        lag[out_of_range] = 0  # set to 0 such that no distance is reduced from sum
        data_animal_id_groups[aid]['lag'] = lag  # each observation has stored distance of its windows first entry (which is then partially reduced)
        data_animal_id_groups[aid]['average_speed'] = data_animal_id_groups[aid]['average_speed'] - \
                                                      ((((data_animal_id_groups[aid].index - timedelta_left) - (data_animal_id_groups[aid].index[ind - 1])) /
                                                      (data_animal_id_groups[aid].index[ind] - data_animal_id_groups[aid].index[ind - 1])) * lag)


        # add last term of sum (as some distance is covered in the time window which is recorded in first observation which is not in window
        ind = data_animal_id_groups[aid].index.searchsorted(data_animal_id_groups[aid].index + timedelta_right, side='right')  # find index of first element not in window
        out_of_range = (ind >= data_animal_id_groups[aid].shape[0])
        ind[out_of_range] = 0  # otherwise key error
        lag = data_animal_id_groups[aid]['distance'].values[ind]
        lag[out_of_range] = 0  # set to 0 such that no distance is added to sum
        data_animal_id_groups[aid]['lag'] = lag  # each observation has stored distance of first observation not in window (which is then partially added)
        data_animal_id_groups[aid]['average_speed'] = data_animal_id_groups[aid]['average_speed'] + \
                                                      ((((data_animal_id_groups[aid].index + timedelta_right) - (data_animal_id_groups[aid].index[ind - 1])) /
                                                        (data_animal_id_groups[aid].index[ind] - data_animal_id_groups[aid].index[ind - 1])) * lag)

        data_animal_id_groups[aid].drop(['lag'], axis=1, inplace=True)

        # divide sum of distances by time (here unit of fps is used)
        data_animal_id_groups[aid]['average_speed'] = data_animal_id_groups[aid]['average_speed'] / float(re.search(r'(\d+)', fps).group(0))

        # reset index
        data_animal_id_groups[aid].reset_index(drop=True, inplace=True)

    return data_animal_id_groups

def compute_average_acceleration(data_animal_id_groups, fps):
    """
    Compute average acceleration of mover based on fps parameter. The formula used for calculating average acceleration is: (Final Speed - Initial Speed) / (Total Time Taken).
    Size of traveling window is determined by fps parameter:
    By choosing f.e. fps=4 at timestamp 5: (speed at timestamp 7 - speed at timestamp 3) / 4.
    By choosing f.e. fps=3 at timestamp 5: (speed at timestamp 6.5 - speed at timestamp 3.5) / 3. (in this case use of interpolation if timestamps 3.5 and 6.5 do not exist.)
    :param data_animal_id_groups: dictionary with 'animal_id' as keys.
    :param fps: integer to define size of window for integer-formatted time or string to define size of window for datetime-formatted time (For possible units refer to:https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.)
    :return: dictionary, including measure for 'average_acceleration'.
    """
    if (isinstance(fps, int) or isinstance(fps, float)):
        fps = str(fps) + 'us'  # as integer time is converted to time for flexible windows so is fps
    for aid in data_animal_id_groups.keys():
        # set index as time to have flexible sized windows
        if is_numeric_dtype(data_animal_id_groups[aid]['time']):
            data_animal_id_groups[aid].index = pd.to_datetime(data_animal_id_groups[aid]['time'], unit='us')
        else:
            data_animal_id_groups[aid].set_index('time', drop=False, inplace=True)

        # add speed of first and last observation in window
        data_animal_id_groups[aid]['start_speed'] = data_animal_id_groups[
            aid]['average_speed'].rolling(min_periods=1, window=fps,
                center=True, closed='both').apply(lambda x: x[0], raw=True).fillna(0)
        data_animal_id_groups[aid]['end_speed'] = data_animal_id_groups[
            aid]['average_speed'].rolling(min_periods=1, window=fps,
                                          center=True, closed='both').apply(lambda x: x[-1], raw=True).fillna(0)

        timedelta = pd.to_timedelta(fps)  # size time window
        timedelta_left = timedelta / 2  # size window left of observation
        timedelta_right = timedelta / 2  # size window right of observation

        # adjust start speed (as first observation prior to first observation in window is taken into account)
        ind = data_animal_id_groups[aid].index.searchsorted(data_animal_id_groups[aid].index - timedelta_left, side='left')  # find index of first element in window
        out_of_range = (ind <= 0)
        lag = data_animal_id_groups[aid]['average_speed'].values[ind-1]  # store respective value of first observation prior to window
        lag[out_of_range] = data_animal_id_groups[aid]['average_speed'][0]  # set to speed of first observation
        data_animal_id_groups[aid]['lag'] = lag  # each observation has stored speed of first element prior to window
        # start speed is calculated as weighted sum of speed from first observation prior to window and from first observation in window
        data_animal_id_groups[aid]['start_speed'] = ((((data_animal_id_groups[aid].index - timedelta_left) - (data_animal_id_groups[aid].index[ind - 1])) /
                    (data_animal_id_groups[aid].index[ind] - data_animal_id_groups[aid].index[ind - 1])) * data_animal_id_groups[aid]['start_speed']) \
                    +((1-(((data_animal_id_groups[aid].index - timedelta_left) - (data_animal_id_groups[aid].index[ind - 1])) /
                    (data_animal_id_groups[aid].index[ind] - data_animal_id_groups[aid].index[ind - 1]))) * lag)


        # adjust end speed (as first observation after window is taken into account)
        ind = data_animal_id_groups[aid].index.searchsorted(data_animal_id_groups[aid].index + timedelta_right, side='right')  # find index of first element not in window
        out_of_range = (ind >= data_animal_id_groups[aid].shape[0])
        ind[out_of_range] = 0  # otherwise key error
        lag = data_animal_id_groups[aid]['average_speed'].values[ind]  # store respective value of first observation not in window
        lag[out_of_range] = data_animal_id_groups[aid]['average_speed'][data_animal_id_groups[aid].shape[0] - 1]  # set to speed of last observation
        data_animal_id_groups[aid]['lag'] = lag  # each observation has stored speed of first observation not in window
        # end speed is calculated as weighted sum of speed from last observation in window and from first observation after window
        data_animal_id_groups[aid]['end_speed'] = ((((data_animal_id_groups[aid].index + timedelta_right) - (data_animal_id_groups[aid].index[ind - 1])) /
                    (data_animal_id_groups[aid].index[ind] - data_animal_id_groups[aid].index[ind - 1])) * lag) + \
                    ((1-(((data_animal_id_groups[aid].index + timedelta_right) - (data_animal_id_groups[aid].index[ind - 1])) /
                    (data_animal_id_groups[aid].index[ind] - data_animal_id_groups[aid].index[ind - 1]))) * data_animal_id_groups[aid]['end_speed'])


        # calculate average acceleration by dividing difference by time (here unit of fps is used)
        data_animal_id_groups[aid]['average_acceleration'] = (data_animal_id_groups[aid]['end_speed'] - data_animal_id_groups[aid]['start_speed'])\
                                                      / float(re.search(r'(\d+)', fps).group(0))

        # reset index and drop columns
        data_animal_id_groups[aid].reset_index(drop=True, inplace=True)
        data_animal_id_groups[aid].drop(['lag', 'start_speed', 'end_speed'], axis=1, inplace=True)

    return data_animal_id_groups


def extract_features(data, fps=10, stop_threshold=0.5):
    """
    Calculate and return all absolute features for input animal group.
    Combined usage of the functions on DataFrame grouping_data(), compute_distance(), compute_direction(), compute_average_speed(),
    compute_average_acceleration(), computing_stops()
    :param data: pandas DataFrame with all records of movements.
    :param fps: size of window used to calculate average speed and average acceleration:
    integer to define size of window for integer-formatted time or string to define size of window for datetime-formatted time (For possible units refer to:https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.)
    :param stop_threshold: integer to specify threshold for average speed, such that we consider timestamp a "stop".
    :return: pandas DataFrame with additional variables consisting of all relevant features.
    """

    if len(data) >= 200000:  # then multiproccessing
        # creating list of different animals df's to split them in pool
        data = grouping_data(data)
        df_list = []
        for aid in data.keys():
            df_list.append(data[aid])

        # use multiprocessing to call extract_features for each animal with different process
        if __name__ == "movekit.feature_extraction":
            pool = multiprocessing.Pool()
            func = partial(extract_features_multiproccessing, fps=fps, stop_threshold=stop_threshold)
            result = pool.map(func, df_list)
            #results = []
            #for result in tqdm(pool.imap(func, df_list), total=len(df_list), desc='Extracting all absolute features'):
            #    results.append(result)

        # regroup in one big data frame and return
        big_df = pd.DataFrame()
        for df in result:
            big_df = big_df.append(df, ignore_index=True)
        big_df = big_df.sort_values(by=['time','animal_id'])
        big_df = big_df.reset_index(drop=True)

        return big_df


    with tqdm(total=100, position=0,
              desc="Extracting all absolute features") as pbar:  # to implement percentage loading bar
        tmp_data = grouping_data(data)
        pbar.update(10)  # first part takes about 10 % of the time
        tmp_data = compute_distance(tmp_data)
        tmp_data = compute_direction(tmp_data,
                                     pbar)  # as computing the direction takes most of the time, the percentage bar is given as a parameter
        tmp_data = compute_turning(tmp_data)
        tmp_data = compute_average_speed(tmp_data, fps)
        tmp_data = compute_average_acceleration(tmp_data, fps)
        tmp_data = computing_stops(tmp_data, stop_threshold)

        # Regroup dictionary into pd DataFrame
        regrouped_data = regrouping_data(tmp_data)

        # Replace NA
        regrouped_data.fillna(0, inplace=True)

        # Put extract features columns to the beginning of df
        cols = regrouped_data.columns.tolist()
        for i in ['time', 'animal_id', 'x', 'y', 'distance', 'direction', 'turning', 'average_speed',
                  'average_acceleration', 'stopped']:
            cols.remove(i)
        cols = ['time', 'animal_id', 'x', 'y', 'distance', 'direction', 'turning', 'average_speed',
                'average_acceleration', 'stopped'] + cols
        regrouped_data = regrouped_data[cols]

        return regrouped_data


def computing_stops(data_animal_id_groups, threshold_speed):
    """
    Calculate absolute feature, describing a record as stop, based on threshold.
    Calculate absolute feature called 'Stopped' where the value is 1 if 'Average_Speed' <= threshold_speed
    and 0 otherwise.
    :param data_animal_id_groups: dictionary with 'animal_id' as keys.
    :param threshold_speed:  integer, defining maximum value for 'average_speed' to be considered as a stop.
    :return: dictionary, including variable 'stopped'.
    """

    for aid in data_animal_id_groups.keys():
        data_animal_id_groups[aid]['stopped'] = np.where(
            data_animal_id_groups[aid]['average_speed'] <= threshold_speed, 1,
            0)
    return data_animal_id_groups


def distance_by_time(data, frm, to):
    """
    Computes the distance between positions for a particular time window for all movers.
    :param data: pandas DataFrame with all records of movements.
    :param frm: int defining the start of the time window. Note that if time is stored as a date (if input data has time not stored as numeric type it is automatically converted to datetime) parameter has to be set using an datetime format: mkit.distance_by_time(data, "2008-01-01", "2010-10-01")
    :param to: Int, defining end point up to where to extract records.
    :param to: int defining the end of the time window (inclusive)
    :return: pandas DataFrame with animal_id and distance
    """
    # use auxiliary functions to get distances for each timestep
    data_animal_id_groups = grouping_data(data)
    data_animal_id_groups = compute_distance(data_animal_id_groups)

    aids = []
    distances = []
    # iterate over grouped dataframes
    for aid in data_animal_id_groups.keys():
        # take a subst of each dataframe as defined by the time window
        df = data_animal_id_groups[aid]
        subset = df[(df['time'] >= frm) & (df['time'] <= to)]
        # sum up all distances in that time window
        distance = subset['distance'].sum()
        aids.append(aid)
        distances.append(distance)

    return pd.DataFrame({'animal_id': aids, 'distance': distances})


def group_movement(feats):
    """
    Returns aggregated movement data, such as distance, mean speed, mean acceleration and mean distance to centroid for the entire group at each time capture.
    :param feats: pd DataFrame with animal-specific data - if no features contained, they will be extracted.
    :return: pd DataFrame with group-specific values for each time-capture

    """

    # Handling no features in input
    if 'distance' not in feats.columns or 'average_speed' not in feats.columns or 'average_acceleration' not in feats.columns:
        warnings.warn(
            'Recalculating features, since distance, speed or acceleration not found in input.'
        )
        feats = extract_features(feats)

    # Handling no centroid in input
    if 'distance_to_centroid' not in feats.columns:
        warnings.warn(
            'Recalculating centroid-distances, since not found in input dataset!'
        )
        feats = centroid_medoid_computation(feats)
    # Group by time, return new dataframe
    data_dist = feats.groupby('time')

    group = pd.DataFrame({
        "total_dist":
            data_dist.sum()['distance'],
        "mean_speed":
            data_dist.mean()['average_speed'],
        "mean_acceleration":
            data_dist.mean()['average_acceleration'],
        "mean_distance_centroid":
            data_dist.mean()['distance_to_centroid']
    })

    return group


def centroid_medoid_computation(data,
                                only_centroid=False,
                                object_output=False):
    """
    Calculates the data point (animal_id) closest to center/centroid/medoid for a time step
    Uses group by on 'time' attribute
    :param data: Pandas DataFrame containing movement records
    :param only_centroid: Boolean in case we just want to compute the centroids. Default: False.
    :param object_output: Boolean whether to create a point object for the calculated centroids. Default: False.
    :return: Pandas DataFrame containing computed medoids & centroids
    """

    # check 3D presence
    if presence_3d(data):
        is_3d = True
    else:
        is_3d = False

    # Group by 'time'-
    data_time = data.groupby('time')

    # Dictionary to hold grouped data by 'time' attribute-
    data_groups_time = {}

    for aid in tqdm(data_time.groups.keys(), position=0, desc="Calculating centroid distances"):
        data_groups_time[aid] = data_time.get_group(aid)
        data_groups_time[aid].reset_index(drop=True, inplace=True)

        # Add 3 additional columns to each group-
        data_l = [0 for x in range(data_groups_time[aid].shape[0])]

        data_groups_time[aid] = data_groups_time[aid].assign(x_centroid=data_l)
        data_groups_time[aid] = data_groups_time[aid].assign(y_centroid=data_l)
        if is_3d:
            data_groups_time[aid] = data_groups_time[aid].assign(z_centroid=data_l)

        if only_centroid == False:
            data_groups_time[aid] = data_groups_time[aid].assign(medoid=data_l)
            data_groups_time[aid] = data_groups_time[aid].assign(
                distance_to_centroid=data_l)

        # Calculate centroid coordinates (x, y)-
        x_mean = np.around(np.mean(data_groups_time[aid]['x']), 3)
        y_mean = np.around(np.mean(data_groups_time[aid]['y']), 3)
        if is_3d:
            z_mean = np.around(np.mean(data_groups_time[aid]['z']), 3)

        data_groups_time[aid] = data_groups_time[aid].assign(x_centroid=x_mean)
        data_groups_time[aid] = data_groups_time[aid].assign(y_centroid=y_mean)
        if is_3d:
            data_groups_time[aid] = data_groups_time[aid].assign(z_centroid=z_mean)

        # add additional variable for point object
        if object_output == True:
            data_groups_time[aid] = data_groups_time[aid].assign(
                centroid=functions.ST_MakePoint(x_mean, y_mean))

        if only_centroid == False:
            # Squared distance of each 'x' coordinate to 'centroid'-
            x_temp = (data_groups_time[aid].loc[:, 'x'] - x_mean) ** 2

            # Squared distance of each 'y' coordinate to 'centroid'-
            y_temp = (data_groups_time[aid].loc[:, 'y'] - y_mean) ** 2

            # Squared distance of each 'z' coordinate to 'centroid'-
            if is_3d:
                z_temp = (data_groups_time[aid].loc[:, 'z'] - z_mean) ** 2

            # Distance of each point from centroid-
            if is_3d:
                dist = np.sqrt(x_temp + y_temp + z_temp)
            else:
                dist = np.sqrt(x_temp + y_temp)

            # Assign computed distances to 'distance_to_centroid' attribute-
            data_groups_time[aid] = data_groups_time[aid].assign(
                distance_to_centroid=np.around(dist, decimals=3))

            # Find 'animal_id' nearest to centroid for this group-
            pos = np.argmin(
                data_groups_time[aid]['distance_to_centroid'].values)
            nearest = data_groups_time[aid].loc[pos, 'animal_id']

            # Assign 'medoid' for this group-
            data_groups_time[aid] = data_groups_time[aid].assign(
                medoid=nearest)

    medoid_data = regrouping_data(data_groups_time)
    return medoid_data


def euclidean_dist(data):
    """
    Compute the euclidean distance between movers for one individual grouped time step using the Scipy 'pdist' and 'squareform' methods.
    :param data: Preprocessed pandas DataFrame with positional record data containing no duplicates.
    :return: pandas DataFrame, including computed euclidean distances.
    """
    if len(data) >= 200000:  # then multiproccessing
        # creating list of different animals df's to split them in pool
        data = timewise_dict(data)
        df_list = []
        for aid in data.keys():
            df_list.append(data[aid])

        # use multiprocessing to call extract_features for each animal with different process
        if __name__ == "movekit.feature_extraction":
            pool = multiprocessing.Pool()
            result = pool.map(euclidean_dist_multiproccessing, df_list)

        # regroup in one big data frame and return
        big_df = pd.DataFrame()
        for df in result:
            big_df = big_df.append(df, ignore_index=True)
        big_df = big_df.sort_values(by=['time','animal_id'])
        big_df = big_df.reset_index(drop=True)

        return big_df


    if presence_3d(data):
        weights = {'x': 1, 'y': 1, 'z': 1}
    else:
        weights = {'x': 1, 'y': 1}
    out = compute_similarity(data, weights)
    return out


def euclidean_dist_multiproccessing(data):
    """
    Compute the euclidean distance between movers for one individual grouped time step using the Scipy 'pdist' and 'squareform' methods.
    :param data: Preprocessed pandas DataFrame with positional record data containing no duplicates.
    :return: pandas DataFrame, including computed euclidean distances.
    """
    if presence_3d(data):
        weights = {'x': 1, 'y': 1, 'z': 1}
    else:
        weights = {'x': 1, 'y': 1}
    out = compute_similarity_multiproccessing(data, weights)
    return out


def compute_similarity(data, weights, p=2):
    """
    Compute positional similarity between animals.
    Computing the positional similarity in a distance matrix according to animal_id for each time step.
    :param data: pandas DataFrame, containing preprocessed movement records.
    :param weights: dictionary, giving variable's weights in weighted distance calculation.
    :param p: integer, giving p-norm for Minkowski, weighted and unweighted. Default: 2.
    :return: pandas DataFrame, including computed similarities.
    """
    w = []  # weight vector
    not_allowed_keys = ['time', 'animal_id']
    df = pd.DataFrame()
    for key in weights:
        if key in data.columns:
            df[key] = data[key]
            w.append(weights[key])


    # compute the distance for each time moment
    df2 = data.groupby('time')
    df3 = pd.DataFrame()  # empty df in which all the data frames containing the distance are merged
    for start in tqdm(df2.groups.keys(), position=0, desc="Computing euclidean distance"):
        groups_df = df2.get_group(start).groupby('time').apply(similarity_computation, w=w,
                                                               p=p)  # calculate distance for each time period
        df3 = pd.concat([df3, groups_df])  # finally all dataframes are merged in df3

    # combine the distance matrix with the data and return
    return pd.merge(data, df3, left_index=True,
                    right_index=True).sort_values(by=['time', 'animal_id'])


def compute_similarity_multiproccessing(data, weights, p=2):
    """
    Compute positional similarity between animals.
    Computing the positional similarity in a distance matrix according to animal_id for each time step.
    :param data: pandas DataFrame, containing preprocessed movement records.
    :param weights: dictionary, giving variable's weights in weighted distance calculation.
    :param p: integer, giving p-norm for Minkowski, weighted and unweighted. Default: 2.
    :return: pandas DataFrame, including computed similarities.
    """
    w = []  # weight vector
    not_allowed_keys = ['time', 'animal_id']
    df = pd.DataFrame()
    for key in weights:
        if key in data.columns:
            df[key] = data[key]
            w.append(weights[key])

    df2 = data.groupby('time')
    df3 = pd.DataFrame()  # empty df in which all the data frames containing the distance are merged
    for start in tqdm(df2.groups.keys(), position=0, desc="Computing euclidean distance", disable=True):
        groups_df = df2.get_group(start).groupby('time').apply(similarity_computation, w=w,
                                                               p=p)  # calculate distance for each time period
        df3 = pd.concat([df3, groups_df])  # finally all dataframes are merged in df3

    # combine the distance matrix with the data and return
    return pd.merge(data, df3, left_index=True,
                    right_index=True).sort_values(by=['time', 'animal_id'])




def similarity_computation(group, w, p):
    """
    Compute similarity between records.
    Compute the Minkowski similarity for one individual grouped time step using the Scipy pdist and squareform methods
    :param group: pandas DataFrame, containing preprocessed movement records.
    :param w: array, consisting of the weight vector.
    :param p: double, applies the respective p-norm for weighted Minkowski.
    :return: pandas DataFrame, including the distances of the records.
    """
    # ids of each animal
    ids = group['animal_id'].tolist()
    # compute and assign the distances for each time step
    return pd.DataFrame(squareform(pdist(group.loc[:, ['x', 'y']], 'minkowski', p=p)),
                        index=group.index,
                        columns=ids)


def ts_all_features(data):
    """
    Perform time series analysis on record data.
    :param data: pandas DataFrame, containing preprocessed movement records and features.
    :return: pandas DataFrame, containing extracted time series features for each id for each feature.
    """

    # Remove the column 'stopped' as it has nominal values and 'direction' as it is a vector and additional columns from Movebank data.
    rm_colm = ['stopped', 'direction','event-id', 'visible', 'location-long',
       'location-lat', 'behavioural-classification', 'comments',
       'study-specific-measurement', 'sensor-type',
       'individual-taxon-canonical-name', 'tag-local-identifier',
       'study-name']
    df = data[data.columns.difference(rm_colm)]

    time_series_features = tsfresh.extract_features(df,
                                                    column_id='animal_id',
                                                    column_sort='time')

    tsfresh.utilities.dataframe_functions.impute(time_series_features)

    return time_series_features


def ts_feature(data, feature):
    """
    Perform time series analysis by extracting specified time series features from record data.
    :param data: pandas DataFrame, containing preprocessed movement records and features.
    :param feature: time series feature which is extracted from the movement records.
    :return: pandas DataFrame, containing defined extracted time series features for each id for each feature.
    """
    fc_parameters = tsfresh.feature_extraction.ComprehensiveFCParameters()
    if feature in fc_parameters:
        settings = {}
        settings[feature] = fc_parameters[feature]

        # Remove the column 'stopped' as it has nominal values and 'direction' as it is a vector and additional columns from Movebank data.
        rm_colm = ['stopped', 'direction','event-id', 'visible', 'location-long',
       'location-lat', 'behavioural-classification', 'comments',
       'study-specific-measurement', 'sensor-type','individual-taxon-canonical-name',
        'tag-local-identifier', 'study-name']
        df = data[data.columns.difference(rm_colm)]
        time_series_features = tsfresh.extract_features(
            df,
            column_id='animal_id',
            column_sort='time',
            default_fc_parameters=settings)
        time_series_features = time_series_features.rename_axis('variable', axis=1)  # rename axis
        time_series_features = time_series_features.rename_axis("id", axis=0)
        return time_series_features
    else:
        print("Time series feature is not known.")
        return


def explore_features(data):
    """
    Show percentage of environment space explored by singular animal.
    Using minumum and maximum of 2-D coordinates, given by 'x' and 'y' features in input DataFrame.
    :param data: pandas DataFrame, containing preprocessed movement records.
    :return: None.
    """
    # Compute global minimum and maximum if user
    # has NOT specified the values-
    x_min = data['x'].min()
    x_max = data['x'].max()

    y_min = data['y'].min()
    y_max = data['y'].max()

    if presence_3d(data):
        is_3d = True
        z_min = data['z'].min()
        z_max = data['z'].max()

    # Group 'data' using 'animal_id' attribute-
    data_groups = grouping_data(data)

    # for each animal, how much of the area has the animal covered?
    # % of space explored by each animal? #
    for aid in data_groups:
        aid_x_min = data_groups[aid]['x'].min()
        aid_x_max = data_groups[aid]['x'].max()

        aid_y_min = data_groups[aid]['y'].min()
        aid_y_max = data_groups[aid]['y'].max()

        print("\nAnimal ID: {0} covered % of area:".format(aid))
        print("x-coordinates: minimum = {0:.2f}% & maximum = {1:.2f}%".format(
            (x_min / aid_x_min) * 100, (aid_x_max / x_max) * 100))

        print("y-coordinates: minimum = {0:.2f}% & maximum = {1:.2f}%".format(
            (y_min / aid_y_min) * 100, (aid_y_max / y_max) * 100))

        if is_3d:
            aid_z_min = data_groups[aid]['z'].min()
            aid_z_max = data_groups[aid]['z'].max()
            print("z-coordinates: minimum = {0:.2f}% & maximum = {1:.2f}%".format(
                (z_min / aid_z_min) * 100, (aid_z_max / z_max) * 100))

    return None


def explore_features_geospatial(preprocessed_data):
    """
    Show exploration of environment space by each animal using 'shapely' package.
    Gives singular descriptions of polygon area covered by each animal and combined.
    Additionally a plot of the respective areas is provided.
    :param preprocessed_data: pandas DataFrame, containing preprocessed movement records.
    :return: None.
    """
    with tqdm(total=100, position=0, desc="Calculating covered areas") as pbar:

        # Create dictionary, grouping records by animal ID as key
        data_groups = grouping_data(preprocessed_data)
        percent_update = 100 / (len(data_groups.keys()) + 1)  # since also group object is created

        # Python dict to hold X-Y coordinates for each animal-
        xy_coord = {}

        # Polygon for singular animal ID
        for aid in data_groups.keys():
            xy_coord[aid] = []

        # Extract position information per animal into list of xy-tuples
        for aid in data_groups.keys():
            for x in range(data_groups[aid].shape[0]):
                temp_tuple = (data_groups[aid].loc[x,
                                                   'x'], data_groups[aid].loc[x,
                                                                              'y'])
                xy_coord[aid].append(temp_tuple)
            pbar.update(percent_update)

        # Polygons for individual animals
        for aid in data_groups.keys():
            poly = Polygon(xy_coord[aid]).convex_hull

            # Compute area of singular polygons and plot
            print(
                "\nArea (polygon) covered by animal ID = {0} is = {1:.2f} sq. units\n"
                    .format(aid, poly.area))
            plt.plot(*poly.exterior.xy)

        # Polygon for collective group
        xy_coord_full = []
        for aid in data_groups.keys():
            for x in range(data_groups[aid].shape[0]):
                temp_tuple = (data_groups[aid].loc[x,
                                                   'x'], data_groups[aid].loc[x,
                                                                              'y'])
                xy_coord_full.append(temp_tuple)

        # Create 'Polygon' object using all coordinates for animal ID combined
        full_poly = Polygon(xy_coord_full).convex_hull

        # Compute area of collective polygon and plot
        print("\nArea (polygon) covered by animals collectively is = ",
              full_poly.area, "sq. units")
        plt.plot(*full_poly.exterior.xy, linewidth=5, color="black")
        plt.show()
        pbar.update(percent_update)
        return None


def outlier_detection(dataset, features=["distance", "average_speed", "average_acceleration",
                                         "stopped", "turning"], remove=False, algorithm="KNN", **kwargs):
    """
    Detect outliers based on different pyod algorithms.
    Note: User may decide on different parameters specific to algorithm chosen.
    :param dataset: Dataframe containing the movement records.
    :param features: list of features to detect outliers upon. Default: ["distance", "average_speed", "average_acceleration",
                                         "stopped", "turning"]
    :param remove: Boolean deciding whether outliers should be removed in returned dataframe. Default: False (outliers are not removed).
    :param algorithm: String defining which algorithm to use for finding the outliers. The following algorithms are available:
                                        "KNN", "ECOD", "COPOD", "ABOD", "MAD", "SOS", "KDE", "Sampling", "GMM", "PCA", "KPCA",
                                         "MCD", "CD", "OCSVM", "LMDD", "LOF", "COF", "CBLOF", "LOCI", "HBOS", "SOD", "ROD",
                                         "IForest", "INNE", "LSCP","LODA", "VAE", "SO_GAAL", "MO_GAAL", "DeepSVDD", "AnoGAN", "ALAD", "R-Graph".
                                         Additional available algorithms: FastABOD: call algorithm="ABOD" with method="fast",
                                         AvgKNN: call algorithm="KNN" with method="mean", MedKNN: call algorithm="KNN" with
                                         method="median". Default algorithm is "KNN". For more information regarding all the
                                         algorithms refer to: https://pyod.readthedocs.io/en/latest/pyod.models.html#
    :param kwargs: Specific to the algorithm additional parameters can be specified. For further information about the
                                         algorithm-specific parameters once again refer to:
                                         https://pyod.readthedocs.io/en/latest/pyod.models.html#
                                         For the default algorithm "KNN" some additional parameters are for example:
                                         contamination - the contamination threshold, n_neighbors - the number of neighbors,
                                         method - which KNN to use, and metric - for distance computation.
    :return: Dataframe containing information for each movement record whether outlier or not.
    """
    if algorithm == "KNN":
        from pyod.models.knn import KNN
        clf = KNN(**kwargs)
    elif algorithm == "ECOD":
        from pyod.models.ecod import ECOD
        clf = ECOD(**kwargs)
    elif algorithm == "ABOD":
        from pyod.models.abod import ABOD
        clf = ABOD(**kwargs)
    elif algorithm == "COPOD":
        from pyod.models.copod import COPOD
        clf = COPOD(**kwargs)
    elif algorithm == "MAD":
        from pyod.models.mad import MAD
        clf = MAD(**kwargs)
    elif algorithm == "SOS":
        from pyod.models.sos import SOS
        clf = SOS(**kwargs)
    elif algorithm == "KDE":
        from pyod.models.kde import KDE
        clf = KDE(**kwargs)
    elif algorithm == "Sampling":
        from pyod.models.sampling import Sampling
        clf = Sampling(**kwargs)
    elif algorithm == "GMM":
        from pyod.models.gmm import GMM
        clf = GMM(**kwargs)
    elif algorithm == "PCA":
        from pyod.models.pca import PCA
        clf = PCA(**kwargs)
    elif algorithm == "KPCA":
        from pyod.models.kpca import KPCA
        clf = KPCA(**kwargs)
    elif algorithm == "MCD":
        from pyod.models.mcd import MCD
        clf = MCD(**kwargs)
    elif algorithm == "CD":
        from pyod.models.cd import CD
        clf = CD(**kwargs)
    elif algorithm == "OCSVM":
        from pyod.models.ocsvm import OCSVM
        clf = OCSVM(**kwargs)
    elif algorithm == "LMDD":
        from pyod.models.lmdd import LMDD
        clf = LMDD(**kwargs)
    elif algorithm == "LOF":
        from pyod.models.lof import LOF
        clf = LOF(**kwargs)
    elif algorithm == "COF":
        from pyod.models.cof import COF
        clf = COF(**kwargs)
    elif algorithm == "CBLOF":
        from pyod.models.cblof import CBLOF
        clf = CBLOF(**kwargs)
    elif algorithm == "LOCI":
        from pyod.models.loci import LOCI
        clf = LOCI(**kwargs)
    elif algorithm == "HBOS":
        from pyod.models.hbos import HBOS
        clf = HBOS(**kwargs)
    elif algorithm == "SOD":
        from pyod.models.sod import SOD
        clf = SOD(**kwargs)
    elif algorithm == "ROD":
        from pyod.models.rod import ROD
        clf = ROD(**kwargs)
    elif algorithm == "IForest":
        from pyod.models.iforest import IForest
        clf = IForest(**kwargs)
    elif algorithm == "INNE":
        from pyod.models.inne import INNE
        clf = INNE(**kwargs)
    elif algorithm == "FB" or algorithm == "FeatureBagging":
        warnings.warn('At the moment this algorithm is not supported by movekit.')
    elif algorithm == "LSCP":
        from pyod.models.lscp import LSCP
        clf = LSCP(**kwargs)
    elif algorithm == "XGBOD":
        warnings.warn('At the moment this algorithm is not supported by movekit.')
    elif algorithm == "LODA":
        from pyod.models.loda import LODA
        clf = LODA(**kwargs)
        from pyod.models.loda import LODA
    elif algorithm == "SUOD":
        warnings.warn('At the moment this algorithm is not supported by movekit.')
    elif algorithm == "AutoEncoder":
        warnings.warn('At the moment this algorithm is not supported by movekit.')
    elif algorithm == "VAE":
        from pyod.models.vae import VAE
        clf = VAE(**kwargs)
    elif algorithm == "SO_GAAL":
        from pyod.models.so_gaal import SO_GAAL
        clf = SO_GAAL(**kwargs)
    elif algorithm == "MO_GAAL":
        from pyod.models.mo_gaal import MO_GAAL
        clf = MO_GAAL(**kwargs)
    elif algorithm == "DeepSVDD":
        from pyod.models.deep_svdd import DeepSVDD
        clf = DeepSVDD(**kwargs)
    elif algorithm == "AnoGAN":
        from pyod.models.anogan import AnoGAN
        clf = AnoGAN(**kwargs)
    elif algorithm == "ALAD":
        from pyod.models.alad import ALAD
        clf = ALAD(**kwargs)
    elif algorithm == "R-Graph" or algorithm == "RGraph":
        from pyod.models.rgraph import RGraph
        clf = RGraph(**kwargs)
    elif algorithm == "LUNAR":
        warnings.warn('At the moment this algorithm is not supported by movekit.')

    inp_data = dataset.loc[:, features]

    clf.fit(inp_data)
    scores_pred = clf.predict(inp_data)

    # avoid overwriting input

    # Inserting column, with 1 if outlier, else 0
    if "outlier" in dataset:
        dataset["outlier"] = scores_pred
    else:
        dataset.insert(2, "outlier", scores_pred)
    if remove:
        dataset = dataset.loc[dataset["outlier"]==0, :]
    return dataset


def split_movement_trajectory(data, stop_threshold=0.5, csv=False):
    """
    Split trajectories of movers in stopping and moving phases.
    :param data: pandas DataFrame containing preprocessed movement records.
    :param stop_threshold: integer to specify threshold for average speed, such that we consider timestamp a "stop".
    :param csv: Boolean, defining if each phase shall be exported locally as singular csv.
    :return: dictionary with animal_id as key and list of individual dataFrames for each movement phase as values.
    """
    if not (set(['distance', 'average_speed', 'average_acceleration', 'direction', 'stopped', 'turning']).issubset(
            data.columns)):
        warnings.warn('Some features are missing and thus will be extracted first. Note that is recommended to use function '
                      'extract_features prior to this function to define values for calculation of average speed and stopping criteria.'
                      'The returned Data Frame should afterwards be used as input for this function.')
        data = extract_features(data, stop_threshold=stop_threshold)
    data_groups = grouping_data(data)
    df_dict = {}

    for aid in data_groups.keys():
        df = data_groups[aid]
        df_dict[aid] = []
        beg = 0
        for i in range(1, len(df)):
            if (df.stopped[i] != df.stopped[i - 1]):
                phase_df = data_groups[aid].iloc[beg:i, :]
                phase_df = phase_df.reset_index(drop=True)
                df_dict[aid].append(phase_df)
                beg = i
        # for the last one
        if (df.stopped[len(df) - 1] == df.stopped[len(df) - 2]):
            phase_df = data_groups[aid].iloc[beg:, :]
            phase_df = phase_df.reset_index(drop=True)
            df_dict[aid].append(phase_df)
        else:
            phase_df = data_groups[aid].iloc[(len(df) - 1):, :]
            phase_df = phase_df.reset_index(drop=True)
            df_dict[aid].append(phase_df)
        print(f' For animal {aid} the trajectory was split in {len(df_dict[aid])} phases of moving and stopping.')

    if csv == True:
        for aid in df_dict.keys():
            for i in range(len(df_dict[aid])):
                df_dict[aid][i].to_csv(f'{aid}_{i}.csv', index=False)

    return df_dict


def movement_stopping_durations(data, stop_threshold=0.5):
    """
    Split trajectories of movers in stopping and moving phases and return the duration of each phase.
    :param data: pandas DataFrame containing preprocessed movement records.
    :param stop_threshold: integer to specify threshold for average speed, such that we consider timestamp a "stop".
    :return: dictionary with animal_id as key and DataFrame with the different phases and their durations as value.
    """
    data = split_movement_trajectory(data, stop_threshold)
    df_dict = {}
    for aid in data.keys():
        df_list = data[aid]
        time_df = pd.DataFrame({'animal_id': aid}, index=[0])
        for index, df in enumerate(df_list):
            try:
                time_diff = df_list[index+1].time[0] - df.time[0]
            except:  # for the last phase
               time_diff = (df.time[len(df)-1] - df.time[0]) + (df.time[0] - df_list[index-1].time[len(df_list[index-1])-1])
            if len(time_df.columns) != 1:  # adjust if only one phase for mover
                time_df[f'Duration of phase {index+1} ({"stopping" if df["stopped"][0] == 1 else "moving"})'] = time_diff
            else:
                time_df[f'Duration of phase {index + 1} ({"stopping" if df["stopped"][0] == 1 else "moving"})'] = len(df)

        df_dict[aid] = time_df
    return df_dict


def hausdorff_distance(data, mover1=None, mover2=None):
    """
    Calculate the Hausdorff-Distance between trajectories of different movers.
    :param data: pandas DataFrame containing movement records.
    :param mover1: animal_id of the first mover if Hausdorff distance is just to be calculated between two movers.
    :param mover2: animal_id of the second mover if Hausdorff distance is just to be calculated between two movers
    :return: Hausdorff distance between two specified movers. If no movers are specified, Hausdorff distance between
    all movers in the data to each other as a Pandas DataFrame.
    """
    presence3d = presence_3d(data)
    data = grouping_data(data)

    if mover1 is None and mover2 is None:

        df = pd.DataFrame()
        for aid1 in data.keys():
            dict = {}
            for aid2 in data.keys():
                if (presence3d):
                    hdf_distance1 = directed_hausdorff(data[aid1][["x", "y", "z"]].values,
                                                       data[aid2][["x", "y", "z"]].values)[0]
                    hdf_distance2 = directed_hausdorff(data[aid2][["x", "y", "z"]].values,
                                                       data[aid1][["x", "y", "z"]].values)[0]
                    hdf_distance = max(hdf_distance1, hdf_distance2)
                else:
                    hdf_distance1 = directed_hausdorff(data[aid1][["x", "y"]].values,
                                                       data[aid2][["x", "y"]].values)[0]
                    hdf_distance2 = directed_hausdorff(data[aid2][["x", "y"]].values,
                                                       data[aid1][["x", "y"]].values)[0]
                    hdf_distance = max(hdf_distance1, hdf_distance2)
                dict[aid2] = hdf_distance
            aid_df = pd.DataFrame(dict, index=[aid1])
            df = df.append(aid_df)

        return df

    else:
        if presence3d:
            hdf_distance1 = directed_hausdorff(data[mover1][["x", "y", "z"]].values,
                                               data[mover2][["x", "y", "z"]].values)[0]
            hdf_distance2 = directed_hausdorff(data[mover2][["x", "y", "z"]].values,
                                               data[mover1][["x", "y", "z"]].values)[0]
            hdf_distance = max(hdf_distance1, hdf_distance2)
        else:
            hdf_distance1 = directed_hausdorff(data[mover1][["x", "y"]].values,
                                               data[mover2][["x", "y"]].values)[0]
            hdf_distance2 = directed_hausdorff(data[mover2][["x", "y"]].values,
                                               data[mover1][["x", "y"]].values)[0]
            hdf_distance = max(hdf_distance1, hdf_distance2)

        return hdf_distance


def extract_features_multiproccessing(data, fps=10, stop_threshold=0.5):
    """
    Calculate and return all absolute features for input animal group.
    Combined usage of the functions on DataFrame grouping_data(), compute_distance(), compute_direction(), compute_average_speed(),
    compute_average_acceleration(), computing_stops()
    :param data: pandas DataFrame with all records of movements.
    :param fps: integer to specify the size of the window examined for calculating average speed and average acceleration.
    :param stop_threshold: integer to specify threshold for average speed, such that we consider timestamp a "stop".
    :return: pandas DataFrame with additional variables consisting of all relevant features.
    """

    with tqdm(total=100, position=0,
              desc="Extracting all absolute features", disable=True) as pbar:  # percentage loading bar silenced in multiproccessing
        tmp_data = grouping_data(data)
        pbar.update(10)  # first part takes about 10 % of the time
        tmp_data = compute_distance(tmp_data)
        tmp_data = compute_direction(tmp_data,
                                     pbar)  # as computing the direction takes most of the time, the percentage bar is given as a parameter
        tmp_data = compute_turning(tmp_data)
        tmp_data = compute_average_speed(tmp_data, fps)
        tmp_data = compute_average_acceleration(tmp_data, fps)
        tmp_data = computing_stops(tmp_data, stop_threshold)

        # Regroup dictionary into pd DataFrame
        regrouped_data = regrouping_data(tmp_data)

        # Replace NA
        regrouped_data.fillna(0, inplace=True)

        # Put extract features columns to the beginning of df
        cols = regrouped_data.columns.tolist()
        for i in ['time', 'animal_id', 'x', 'y', 'distance', 'direction', 'turning', 'average_speed',
                  'average_acceleration', 'stopped']:
            cols.remove(i)
        cols = ['time', 'animal_id', 'x', 'y', 'distance', 'direction', 'turning', 'average_speed',
                'average_acceleration', 'stopped'] + cols
        regrouped_data = regrouped_data[cols]

        return regrouped_data


def segment_data(data, feature, threshold, csv=False, fps=10, stop_threshold=0.5):
    """
    Segment data in subsets by feature values using a given threshold value. For instance, by using the average speed as feature split the dataset in segments above and below a given threshold.
    :param data: dataframe containing the feature which is used to split the dataset. Note that if feature is 'distance', 'average_speed', 'average_acceleration', 'direction', 'stopped' or 'turning',
    feature can also be extracted within the function. In that case one should define the input parameters to use when extract_features() is called.
    :param feature: column name of the feature used to split data in subsets.
    :param threshold: threshold used to split data according to feature value.
    :param csv: Boolean, defining if each subset shall be exported locally as singular csv.
    :param fps: used if features are not extracted before but within the function by calling extract_features():
    size of window used to calculate average speed and average acceleration:
    integer to define size of window for integer-formatted time or string to define size of window for datetime-formatted time (For possible units refer to:https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.)
    :param stop_threshold: used if features are not extracted before but within the function by calling extract_features():
    integer to specify threshold for average speed, such that we consider timestamp a "stop".
    :return: dictionary with id of different movers as key and a list of all the subsets for this mover as values. Subsets are thereby stored as dataframe.
    """
    if feature not in data.columns:
        if feature not in ['distance', 'average_speed', 'average_acceleration', 'direction', 'stopped', 'turning']:
            warnings.warn('Given feature to segment data is not a column name of the data. Please redefine the feature or add feature to the data.')
        else:
            warnings.warn('Feature to segment data is missing and thus will be extracted first. Note that it is recommended to use function '
                          'extract_features prior to this function to define values for calculation of average speed and stopping criteria.'
                          'The returned Data Frame should afterwards be used as input for this function.')
            data = extract_features(data, fps=fps, stop_threshold=stop_threshold)

    data_groups = grouping_data(data)
    df_dict = {}

    for aid in data_groups.keys():
        df = data_groups[aid]
        df_dict[aid] = []
        beg = 0
        for i in range(1, len(df)):
            if (df[feature][i] <= threshold and df[feature][i-1]>threshold):
                phase_df = data_groups[aid].iloc[beg:i, :]
                phase_df = phase_df.reset_index(drop=True)
                df_dict[aid].append(phase_df)
                beg = i
            elif (df[feature][i] > threshold and df[feature][i-1]<=threshold):
                phase_df = data_groups[aid].iloc[beg:i, :]
                phase_df = phase_df.reset_index(drop=True)
                df_dict[aid].append(phase_df)
                beg = i
        # for the last one
        if ((df[feature][len(df) - 1] > threshold and df[feature][len(df) - 2] > threshold) or
                (df[feature][len(df) - 1] <= threshold and df[feature][len(df) - 2] <= threshold)):
            phase_df = data_groups[aid].iloc[beg:, :]
            phase_df = phase_df.reset_index(drop=True)
            df_dict[aid].append(phase_df)
        else:
            phase_df = data_groups[aid].iloc[(len(df) - 1):, :]
            phase_df = phase_df.reset_index(drop=True)
            df_dict[aid].append(phase_df)
        print(f' For animal {aid} the trajectory was split in {len(df_dict[aid])} phases.')

    if csv == True:
        for aid in df_dict.keys():
            for i in range(len(df_dict[aid])):
                df_dict[aid][i].to_csv(f'{aid}_{i}.csv', index=False)

    return df_dict


def outlier_by_threshold(data, feature_thresholds, remove=False):
    """
    Identify outliers by user given features with specific minimum and maximum thresholds.
    :param data: data on which outliers are detected.
    :param feature_thresholds: dictionary containing the features as keys and the minimum/maximum threshold as two element list.
    For example if one would only want to declare all data points having an average speed < 5 and > 10 as outliers:
    feature_threshold = {'average_speed': [5,10]}
    :param remove: Boolean deciding whether outliers should be removed in returned dataframe. Default: False (outliers are not removed).
    :return: Dataframe containing information for each record whether it is an outlier according to the defined threshold values.
    """
    data['outlier_by_threshold'] = np.zeros(len(data))

    for k in feature_thresholds.keys():
        data.loc[(data[k] < feature_thresholds[k][0]) | (data[k] > feature_thresholds[k][1]), 'outlier_by_threshold'] = 1

    if remove:
        data = data.loc[data['outlier_by_threshold'] == 0, :]

    data['outlier_by_threshold'] = pd.to_numeric(data['outlier_by_threshold'], downcast='integer')
    return data


def getis_ord(data, x_grids_per_t=3, y_grids_per_t=3, time_grids=3):
    """
    Calculate the Getis-Ord G* statistic for each x-y-time interval of the data. Interval size is specified by input.
    For more information about how the statistic is calculated please refer to: https://sigspatial2016.sigspatial.org/giscup2016/problem
    :param data: pandas Data frame containing the movement data in the columns x, y and time.
    :param x_grids_per_t: int defining how many x intervals there are for each time step. The x axis is subdivided uniformly,
    i.e. if the maximum value of x in the data is 100 and the minimum value is 10, by setting x_grids_per_t = 3 for each
    time step there are 3 intervals ([10,40),[40,70),[70,100])
    :param y_grids_per_t: int defining how many y intervals there are for each time step. The y axis is subdivided uniformly,
    i.e. if the maximum value of y in the data is 50 and the minimum value is 10, by setting y_grids_per_t = 4 for each
    time step there are 4 intervals ([10,20),[20,30),[30,40),[50,50])
    :param time_grids: int defining how many time intervals there are. The time axis is subdivided uniformly, i.e. if
    the maximum value of time in the data is 500 and the minimum value is 0, by setting time_grids = 5 there are 5 time
    intervals ([0,100),[100,200),[200,300),[300,400),[400,500])
    Note that if one defines f.e. x_grids_per_t = 3, y_grids_per_t = 3 and time_grids = 5 the space time cube used for
    calculating G* contains 3*3*5=45 intervals.
    return: Pandas data frame containing the Getis-Ord statistic for each examined interval (intervals are defined by six
    columns defining the respective start and end values of the intervals' x-coordinate, y-coordinate and time.
    """
    # calculate interval values
    time_range = data['time'].max() - data['time'].min()
    time_min = data['time'].min()
    x_range = data['x'].max() - data['x'].min()
    x_min = data['x'].min()
    y_range = data['y'].max() - data['y'].min()
    y_min = data['y'].min()
    int_length_x = x_range / x_grids_per_t
    int_length_y = y_range / y_grids_per_t
    int_length_t = time_range / time_grids
    points = []
    grid = []
    for x in range(x_grids_per_t):
        for y in range(y_grids_per_t):
            for t in range(time_grids):
                points.append((x, y, t))
                grid.append(([x_min + x * int_length_x, x_min + (x+1) * int_length_x],
                      [y_min + y * int_length_y, y_min + (y+1) * int_length_y],
                      [time_min + t * int_length_t, time_min + (t+1) * int_length_t]))  # interval values for x, y, t
    # assign observations to intervals
    df = data[['x', 'y', 'time']]
    scores = np.zeros(len(grid))
    for r in range(len(data)):
        for ind, i in enumerate(grid):
            if (df.iloc[r, 0] >= i[0][0]) and (df.iloc[r, 0] < i[0][1]) and (df.iloc[r, 1] >= i[1][0]) and (df.iloc[r, 1] < i[1][1]) and (df.iloc[r, 2] >= i[2][0]) and (df.iloc[r, 2] < i[2][1]):
                scores[ind] = scores[ind] + 1
                break
            elif (df.iloc[r, 0] == data['x'].max()) or (df.iloc[r, 1] == data['y'].max()) or (df.iloc[r, 2] == data['time'].max()):
                if (df.iloc[r, 0] >= i[0][0]) and (df.iloc[r, 0] <= i[0][1]) and (df.iloc[r, 1] >= i[1][0]) and (
                        df.iloc[r, 1] <= i[1][1]) and (df.iloc[r, 2] >= i[2][0]) and (df.iloc[r, 2] <= i[2][1]):
                    scores[ind] = scores[ind] + 1  # observations having end values of last interval as value
                    break

    # calculate weights in space time cube
    d = squareform(pdist(np.array(points)))
    d[d <= 1.9] = 1
    d[d > 1] = 0
    for i in range(len(d)):
        sum = d[i].sum()
        d[i][d[i] == 1] = sum / len(d[i])

    # calculate G*
    n = len(scores)
    nom_one = d @ scores  # first term nominator
    x_mean = np.mean(scores)
    nom_two = x_mean * np.sum(d, axis=1)  # second term nominator
    S = np.sqrt(((scores ** 2).sum()) / n - x_mean ** 2)  # S
    denom_root = np.sqrt((n * np.sum(d ** 2, axis=1) - (np.sum(d, axis=1)) ** 2) / (n - 1))  # denominator root
    nom = nom_one - nom_two
    denom = S * denom_root
    G = nom / denom

    # return g scores in data frame
    score_df = pd.DataFrame({'time0': np.unique(grid, axis=0)[:, 2, 0].tolist(),
              'time1': np.unique(grid, axis=0)[:, 2, 1].tolist(),
              'x0': np.unique(grid, axis=0)[:, 0, 0].tolist(),
              'x1': np.unique(grid, axis=0)[:, 0, 1].tolist(),
              'y0': np.unique(grid, axis=0)[:, 1, 0].tolist(),
              'y1': np.unique(grid, axis=0)[:, 1, 1].tolist(),
              'Getis-Ord Score': G})
    score_df.sort_values(['time0', 'x0', 'y0'], inplace=True)

    return score_df.reset_index(drop=True)
