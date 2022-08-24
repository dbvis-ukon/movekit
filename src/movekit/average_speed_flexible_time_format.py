import re
from src.movekit.io import *
from src.movekit.clustering import *
import pandas as pd
from pandas.api.types import is_numeric_dtype


def compute_average_speed(data_animal_id_groups, fps):
    if (isinstance(fps, int) or isinstance(fps, float)):
        fps = str(fps) + 'ns'  # as integer time is converted to time for flexible windows so is fps
    for aid in data_animal_id_groups.keys():
        # set index as time to have flexible sized windows
        if is_numeric_dtype(data_animal_id_groups[aid]['time']):
            data_animal_id_groups[aid].index = pd.to_datetime(data_animal_id_groups[aid]['time'])
        else:
            data_animal_id_groups[aid].set_index('time', drop=False, inplace=True)

        # add distances of window (will later be divided by time)
        data_animal_id_groups[aid]['average_speed'] = data_animal_id_groups[
            aid]['distance'].rolling(min_periods=1, window=fps,
                                     center=True, closed='both').sum().fillna(0)

        timedelta = pd.to_timedelta(fps)  # size time window
        # as integers are mapped to nanoseconds factor 10 is applied to not loose information when rounding
        #timedelta_left = (((10*timedelta / 2).ceil('10'+re.search(r'(\D+)', fps).group(0))) - pd.to_timedelta('10'+re.search(r'(\D+)', fps).group(0)))/10  # size window left of observation
        #timedelta_right = ((10*timedelta / 2).floor('10'+re.search(r'(\D+)', fps).group(0)))/10  # size window right of observation

        # here still to fix: integers shouldn't be converted to ns but to larger unit, otherwise /2 will lead to information loss
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
                                                      ((((data_animal_id_groups[aid].index + timedelta_right) - (
                                                      data_animal_id_groups[aid].index[ind - 1])) /
                                                        (data_animal_id_groups[aid].index[ind] -
                                                         data_animal_id_groups[aid].index[ind - 1])) * lag)

        data_animal_id_groups[aid].drop(['lag'], axis=1, inplace=True)

        # divide sum of distances by time (here unit of fps is used)
        data_animal_id_groups[aid]['average_speed'] = data_animal_id_groups[aid]['average_speed'] / float(re.search(r'(\d+)', fps).group(0))


    return data_animal_id_groups


# time format
data = read_data("examples/datasets/fish-time-format.xlsx")
data_grouped = grouping_data(data)
data_grouped = compute_distance(data_grouped)
df = compute_average_speed(data_grouped, fps='4d')
check_df = df[905].loc[:,['distance','average_speed']]

# integer time
data = read_data("examples/datasets/fish-5.csv")
data_grouped = grouping_data(data)
data_grouped = compute_distance(data_grouped)
df = compute_average_speed(data_grouped, fps=4)
check_df = df[905].loc[:,['distance','average_speed']]
