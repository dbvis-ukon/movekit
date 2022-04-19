import pandas as pd
import numpy as np


def presence_3d(data):
    """
    Check whether data is 3-dimensional.

    :param data: pandas Dataframe containing the movement records.
    :return: boolean whether column z in Dataframe.
    """
    return 'z' in data.columns


def angle(vec1, vec2):
    """
    Calculate angle between two vectors
    :param vec1: vector 1
    :param vec2: vector 2
    return: angle in degrees
    """
    cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # circumvent float precision errors. limit to [-1,1] with this check
    if cos > 1:
        cos = 1
    elif cos < -1:
        cos = -1
    return np.rad2deg(np.arccos(cos))

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # circumvent float precision errors. limit to [-1,1] with this check
    if cos > 1:
        cos = 1
    elif cos < -1:
        cos = -1
    elif np.isnan(cos):  # in case one of the given vectors is zero-vector
        cos = 0
    return cos
