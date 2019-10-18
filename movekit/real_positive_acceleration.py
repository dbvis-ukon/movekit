

import pandas as pd
import numpy as np
# import math


def real_positive_acceleration(grouped_data):
    '''
    Function to calculate only positive acceleration

    Input: 'grouped_data' is a Python 3 dictionary containing as key
    'animal_id' and as value, the Pandas DataFrame for that animal ID

    Output: Adds a new attribute viz., 'real_positive_acceleration'
    where, if acceleration is positive, the calculated value is used
    and if otherwise, 0 value is used
    Returns a single concatenated Pandas DataFrame
    '''

    for aid in grouped_data.keys():
        print("\nCurrent animal_id being calculated for is: {0}".format(aid))

        for i in range(0, grouped_data[aid].shape[0]):
            if grouped_data[aid].loc[i, 'average_acceleration'] < 0:
                grouped_data[aid].loc[i, 'real_positive_acceleration'] = 0
            else:
                grouped_data[aid].loc[i, 'real_positive_acceleration'] = grouped_data[aid].loc[i, 'average_acceleration'] 

    # return grouped_data

    # Concatenate all Pandas DataFrame of Python 3 dictionary into one-
    result = pd.concat(grouped_data[aid] for aid in grouped_data.keys())

    # Reset index-
    result.reset_index(drop=True, inplace=True)

    return result




# Example usage:

"""
# Use the following functions from 'absolute.py' file in 'feature_extraction' folder-

# Read in CSV file-
data = pd.read_csv("fish-5.csv")

# Group data according to 'animal_id' attribute of CSV data-
grouped_data = grouping_data(data)

# Compute distance and direction using the function from 'absolute.py' file-
dist_direction = compute_distance_and_direction(grouped_data)

# Compute average speed using the function from 'absolute.py' file-
# NOTE: Here, fps = 5
avg_speed_fps_5 = compute_average_speed(dist_direction, fps=5)

# Compute average acceleration using the function from 'absolute.py' file-
# NOTE: Here, fps = 5
avg_acc_fps_5 = compute_average_acceleration(avg_speed_fps_5, fps=5)
"""

# Compute real positive acceleration-
# NOTE: 'fps' is and has to be same as above as the computations are dependent on each other
pos_acc = real_positive_acceleration(avg_acc_fps_5)

# Optional-
# Write result to HDD-
# pos_acc.to_csv("real_positive_acceleration_fps_5.csv", index=False)

