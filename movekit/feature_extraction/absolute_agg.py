"""
  Extract aggregated absolute features - e.g. traveled distance for one animal
  Author: Arjun Majumdar, Eren Cakmak
  Created: September, 2019
"""

import pandas as pd
import numpy as np
import math


def compute_distance_summary(data_featrues):
    '''
    Function to calculate the sum of metric distance travelled by an animal.
    Also calculates the maximum distance travelled by an animal.

    Input: Accepts a Python 3 dictionary containing animal_id as key and its
    Pandas Data Frame as value

    Returns: A Python 3 dictionary containing two nested dictionaries computing-
    sum of total metric distance travelled by each animal
    maximum distance travelled by each animal
    '''
    result = {}

    # doing a new grouping
    for index, group in data_featrues.groupby('animal_id'):
        # compute the results
        result[index] = {
            'sum_of_distance': group['distance'].sum(),
            'maximum_distance': group['distance'].max()}

        print("\nanimal_id = {0} travelled the distance = {1}".format(
            index, result[index]['sum_of_distance']))
        print("\nanimal_id = {0} travelled the max distance between two time steps = {1}".format(
            index, result[index]['maximum_distance']))

    return result


def compute_stop_summary(data_featrues):
    '''
    Function to compute how long/time steps, each animal (animal_id)
    is in motion and is stationary
    This is done for each 'animal_id'

    Input: Python 3 dictionary containing 'Stopped' attribute from using 'computing_stops()' function
    Returns: Textual description per animal_id of the number of time steps for which they were
    in motion and were stationary
    '''
    result = {}

    # doing a grouping
    for index, group in data_featrues.groupby('animal_id'):
        result[index] = {
            'stopped': group['stopped'].eq(1).sum(),
            'moving': group['stopped'].eq(0).sum()}

        print("\nanimal_id = {0} is in motion for = {1} time steps".format(
            index, result[index]['stopped']))
        print("animal_id = {0} is stationary for = {1} time steps\n".format(
            index, result[index]['moving']))

    return result
