"""
  Extract comparative features - features which describe relations between movers
  Author: Arjun Majumdar, Eren Cakmak
  Created: September, 2019
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


import movekit.io
import movekit.preprocess
import movekit.feature_extraction


def distance_euclidean_matrix(data):
    """
    A function to create a distance matrix according to animal_id for each
    time step

    Input: Pandas Data Frame containing CSV file
    Output: Pandas Data Frame having distance matrix created by function

    example usage
    distance_matrix = distance_euclidean_matrix(data)
    """
    return data.groupby('time').apply(euclidean_dist).sort_values(by=['time', 'animal_id'])


def euclidean_dist(group):
    """
    Compute the distance for one individual grouped time step using the 
    Scipy pdist and squareform methods 
    """
    # ids of each animal
    ids = group['animal_id'].tolist()
    # compute and assign the distances for each time step
    group[ids] = pd.DataFrame(squareform(
        pdist(group[['x', 'y']], 'euclidean')),
        index=group.index, columns=ids)
    return group


if __name__ == "__main__":
    path_to_file = "examples/datasets/fish-5.csv"
    # Read in CSV file using 'path_to_file' variable-
    data = movekit.io.parse_csv(path_to_file)
    distance_data = distance_euclidean_matrix(data)
    print(distance_data)
