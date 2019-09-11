"""
  Extract comparative features - features which describe relations between movers
  Author: Arjun Majumdar, Eren Cakmak
  Created: September, 2019
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix


def distance_euclidean_matrix(data):
    """
    A function to create a distance matrix according to animal_id for each
    time step

    Input: Pandas Data Frame containing CSV file
    Output: Pandas Data Frame having distance matrix created by function

    example usage
    distance_matrix = distance_euclidean_matrix(data)

    Write file to HDD (optional)-
    distance_matrix.to_csv("fish-5_processed.csv", index=False)
    """

    # Group by 'time' attribute-
    data_time = {}

    groups = data.groupby('time')

    for time in groups.groups.keys():
        data_time[time] = groups.get_group(time)

    # Reset index-
    for time in data_time.keys():
        data_time[time].reset_index(drop=True, inplace=True)

    # distance_matrix(data_time[1].loc[:, ['x', 'y']].values, data_time[1].loc[:, ['x', 'y']])

    final_matrix = {}

    for time in data_time.keys():
        final_matrix[time] = pd.DataFrame(
            distance_matrix(data_time[1].loc[:, ['x', 'y']].values,
                            data_time[1].loc[:, ['x', 'y']]),
            index=data_time[1].loc[:, 'animal_id'].values,
            columns=data_time[1].loc[:, 'animal_id'].values)

    # Concatenate different groups into one Pandas DataFrame-
    result = pd.concat(final_matrix[time] for time in final_matrix.keys())

    # Save index in 'first_col' attribute-
    first_col = result.index

    # Add this as 'first_column' attribute-
    result['animal_id'] = first_col
    """
	time_step = np.repeat(np.arange(1, data['time'].max()), 5)
	result['time_step'] = time_step
	"""

    cols = result.columns.tolist()

    # Re-arrange columns-
    cols = cols[-1:] + cols[:-1]

    result = result[cols]

    # Reset indices-
    result.reset_index(drop=True, inplace=True)

    return result
