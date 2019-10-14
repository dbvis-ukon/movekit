"""
  Extract comparative features - features which describe relations between movers
  Author: Arjun Majumdar, Eren Cakmak
  Created: September, 2019
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing


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


def compute_similarity(data, weights, p=2):
    """
    A function to compute the similarity between animals in a distance matrix according to animal_id for each time step

    Input: Pandas Data Frame containing CSV file
    weights = dictonary giving the specifc variables weights in the weighted distance calculation
    p : scalar The p-norm to apply for Minkowski, weighted and unweighted. Default: 2.
    Output: Pandas Data Frame having distance matrix created by function
    """
    w = []  # weight vector
    not_allowed_keys = ['time', 'animal_id']
    df = pd.DataFrame()
    for key in weights:
        if key in data.columns:
            df[key] = data[key]
            w.append(weights[key])
    # normalize the data frame
    normalized_df = (df-df.min())/(df.max()-df.min())
    # add the columns time and animal id to the window needed for group by and the column generation
    normalized_df[not_allowed_keys] = data[not_allowed_keys]
    # compute the distance for each time moment
    df2 = normalized_df.groupby('time').apply(similarity_computation, w=w, p=p)
    # combine the distance matrix with the data and return
    return pd.merge(data, df2, left_index=True, right_index=True).sort_values(by=['time', 'animal_id'])


def similarity_computation(group, w, p):
    """
    Compute the minkowski similarity for one individual grouped time step using the Scipy pdist and squareform methods
    """
    # ids of each animal
    ids = group['animal_id'].tolist()
    # compute and assign the distances for each time step
    return pd.DataFrame(squareform(pdist(group, 'wminkowski', p=p, w=w)),
                        index=group.index, columns=ids)


# Usage example
# if __name__ == "__main__":
#     path_to_file = "examples/datasets/fish-5.csv"
#     # Read in CSV file using 'path_to_file' variable-
#     data = movekit.io.parse_csv(path_to_file)
#     data_grouped = movekit.preprocess.grouping_data(data)
#     data_features = movekit.feature_extraction.compute_absolute_features(
#         data_grouped)
#     # print(data_features)

#     weights = {'Distance': 1,  'Average_Speed': 1,
#                'Average_Acceleration': 1, 'x': 1, 'y': 1}
#     result_data = compute_similarity(data_features, weights)

#     print(result_data)
