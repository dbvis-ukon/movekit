

import pandas as pd
import numpy as np
import math


def feature_moving_summary(data_grouped_stops):
    '''
    Function to compute how long/time steps, each animal (animal_id)
    is in motion and is stationary
    This is done for each 'animal_id'

    Input: Python 3 dictionary containing 'Stopped' attribute from using 'computing_stops()' function
    Returns: Textual description per animal_id of the number of time steps for which they were
    in motion and were stationary
    '''

    for aid in data_grouped_stops.keys():
        print("\nanimal_id = {0} is in motion for = {1} time steps".format(aid, data_grouped_stops[aid]['Stopped'].eq('yes').sum()))
        print("animal_id = {0} is stationary for = {1} time steps\n".format(aid, data_grouped_stops[aid]['Stopped'].eq('no').sum()))

    return None


# Example usage-

# Read in CSV file-
# data = pd.read_csv("/path_to_file/fish-5.csv")

# Use the following functions from 'absolute.py' file in 'feature_extraction' folder-

# Group data according to 'animal_id'-
# grouped_data = grouping_data(data)

# Compute distance-
# distance_data = compute_distance(grouped_data)

# Compute average speed-
# average_speed = compute_average_speed(distance_data, fps = 3)

# Compute stops-
# stops_data = computing_stops(average_speed, threshold_speed= 0.9)

# Get animal moving and stop summary-
# feature_moving_summary(stops_data)


