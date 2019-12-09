import math
import pandas as pd
import numpy as np
from tsfresh import extract_features
from scipy.spatial.distance import pdist, squareform
from tsfresh.utilities.dataframe_functions import impute


def compute_distance_summary(data_featrues):

    """
    Function to calculate the sum of metric distance travelled by an animal.
    Also calculates the maximum distance travelled by an animal.

    Input: Accepts a Python 3 dictionary containing animal_id as key and its
    Pandas Data Frame as value

    Returns: A Python 3 dictionary containing two nested dictionaries computing-
    sum of total metric distance travelled by each animal
    maximum distance travelled by each animal
    """

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

    """
    Function to compute how long/time steps, each animal (animal_id)
    is in motion and is stationary
    This is done for each 'animal_id'

    Input: Python 3 dictionary containing 'Stopped' attribute from using 'computing_stops()' function
    Returns: Textual description per animal_id of the number of time steps for which they were
    in motion and were stationary
    """

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


def grouping_data(processed_data):

    """
    A function to group all values for each 'animal_id' attribute

    Input is 'processed_data' which is processed Pandas DataFrame
    Returns a dictionary where- key is animal_id & value is Pandas DataFrame
    for that 'animal_id'
    """

    # A dictionary object to hold all groups obtained using group by-
    data_animal_id_groups = {}

    # Group by using 'animal_id' attribute-
    data_animal_id = processed_data.groupby('animal_id')

    # Get each animal_id's data from grouping performed-
    for animal_id in data_animal_id.groups.keys():
        data_animal_id_groups[animal_id] = data_animal_id.get_group(animal_id)

    # To reset index for each group-
    for animal_id in data_animal_id_groups.keys():
        data_animal_id_groups[animal_id].reset_index(drop=True, inplace=True)

    # Add additional attributes/columns to each groups-
    for aid in data_animal_id_groups.keys():
        data = [0 for x in range(data_animal_id_groups[aid].shape[0])]

        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            distance=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            average_speed=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            average_acceleration=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            positive_acceleration=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            direction=data)

    return data_animal_id_groups


def compute_absolute_features(data_animal_id_groups, fps=10, stop_threshold=0.5):

    """Calculate absolute features for the data animal group"""

    direction_distance_data = compute_distance_and_direction(
        data_animal_id_groups)

    avg_speed_data = compute_average_speed(direction_distance_data, fps)
    avg_acceleration_data = compute_average_acceleration(avg_speed_data, fps)
    stop_data = computing_stops(avg_acceleration_data, stop_threshold)

    return stop_data


def computing_stops(data_animal_id_groups, threshold_speed):

    """
    Calculate absolute feature called 'Stopped' where the value is 'yes' if
    'Average_Speed' <= threshold_speed and 'no' otherwise
    """

    data_animal_id_groups['stopped'] = np.where(
        data_animal_id_groups['average_speed'] <= threshold_speed, 1, 0)

    print(
        "\nNumber of movers stopped according to threshold speed = {0} is {1}".
        format(threshold_speed, data_animal_id_groups['stopped'].eq(1).sum()))

    print(
        "Number of movers moving according to threshold speed = {0} is {1}\n".
        format(threshold_speed, data_animal_id_groups['stopped'].eq(0).sum()))

    return data_animal_id_groups


def compute_distance_and_direction(data_animal_id_groups):

    """
    Calculate metric distance and direction-

    Calculate the metric distance between two consecutive time frames/time stamps
    for each moving entity (in this case, fish)
    """

    for aid in data_animal_id_groups.keys():
        print("\nComputing Distance & Direction for Animal ID = {0}\n".format(
            aid))

        # for i in range(1, animal_id.shape[0] - 1):
        for i in range(1, data_animal_id_groups[aid].shape[0] - 1):
            # print("Current i = ", i)

            x1 = data_animal_id_groups[aid].iloc[i, 2]
            y1 = data_animal_id_groups[aid].iloc[i, 3]
            x2 = data_animal_id_groups[aid].iloc[i + 1, 2]
            y2 = data_animal_id_groups[aid].iloc[i + 1, 3]

            # Compute distance between 2 points-
            distance = math.sqrt(
                math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))

            # Compute the direction in DEGREES-
            direction = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
            if math.isnan(direction):
                data_animal_id_groups[aid].loc[i, 'direction'] = 0
            else:
                data_animal_id_groups[aid].loc[i, 'direction'] = direction

            # Insert computed distance to column/attribute 'Distance'-
            data_animal_id_groups[aid].loc[i, 'distance'] = distance

    return data_animal_id_groups


def compute_average_speed(data_animal_id_groups, fps):

    """
    Average Speed-

    A function to compute average speed of an animal based on fps
    (frames per second) parameter. Calculate the average speed of a mover,
    based on the pandas dataframe and a frames per second (fps) parameter

    Formula used-
    Average Speed = Total Distance Travelled / Total Time taken
    """

    # start_time = time.time()

    for aid in data_animal_id_groups.keys():
        print("\nComputing Average Speed for Animal ID = {0}\n".format(aid))

        # for i in range (1, animal_id.shape[0] - fps + 1):
        for i in range(1, data_animal_id_groups[aid].shape[0] - fps + 1):

            tot_dist = 0  # total distance travelled

            for j in range(i, i + fps):
                # tot_dist += animal_id.loc[j, "Distance"]
                tot_dist += data_animal_id_groups[aid].loc[j, "distance"]

            # animal_id.loc[i, "Average_Speed"] = (tot_dist / fps)
            data_animal_id_groups[aid].loc[i, "average_speed"] = (tot_dist /
                                                                  fps)

    return data_animal_id_groups


def compute_average_acceleration(data_animal_id_groups, fps):

    """
    A function to compute average acceleration of an animal based on fps
    parameter.
    """

    for aid in data_animal_id_groups.keys():
        print("\nComputing Average Speed for Animal ID = {0}\n".format(aid))

        # for i in range (1, animal_id.shape[0] - fps + 1):
        for i in range(1, data_animal_id_groups[aid].shape[0] - fps + 1):
            avg_speed = 0

            # Calculating Average Speed-
            avg_speed = data_animal_id_groups[aid].loc[i, 'average_speed'] - \
                data_animal_id_groups[aid].loc[i + 1, 'average_speed']
            # compute acceleration
            data_animal_id_groups[aid].loc[i, 'average_acceleration'] = (
                avg_speed / fps)

            # postive acceleration
            if data_animal_id_groups[aid].loc[i, 'average_acceleration'] <= 0:
                data_animal_id_groups[aid].loc[i,
                                               'real_positive_acceleration'] = 0
            else:
                data_animal_id_groups[aid].loc[i,
                                               'real_positive_acceleration'] = data_animal_id_groups[aid].loc[i, 'average_acceleration']
            # NaN values of real postive acceleration to zero
            data_animal_id_groups[aid]['real_positive_acceleration'].fillna(
                0, inplace=True)
    # Concatenate all Pandas DataFrame into one-
    result = pd.concat(data_animal_id_groups[aid]
                       for aid in data_animal_id_groups.keys())

    # Reset index-
    result.reset_index(drop=True, inplace=True)

    return result


def time_series_analyis(data):

    # remove the columns stopped as it has nominal values
    rm_colm = ['stopped']
    df = data[data.columns.difference(rm_colm)]
    extracted_features = extract_features(df,
                                          column_id='animal_id',
                                          column_sort='time')
    impute(extracted_features)

    return extracted_features


def compute_distance_summary(data_featrues):

    """
    Function to calculate the sum of metric distance travelled by an animal.
    Also calculates the maximum distance travelled by an animal.

    Input: Accepts a Python 3 dictionary containing animal_id as key and its
    Pandas Data Frame as value

    Returns: A Python 3 dictionary containing two nested dictionaries computing-
    sum of total metric distance travelled by each animal
    maximum distance travelled by each animal
    """

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

    """
    Function to compute how long/time steps, each animal (animal_id)
    is in motion and is stationary
    This is done for each 'animal_id'

    Input: Python 3 dictionary containing 'Stopped' attribute from using 'computing_stops()' function
    Returns: Textual description per animal_id of the number of time steps for which they were
    in motion and were stationary
    """

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
    A function to compute the similarity between animals in a distance matrix
    according to animal_id for each time step.

    data : Pandas Data Frame containing CSV file
    weights : dictonary giving the specifc variables weights in the weighted
        distance calculation
    p : scalar The p-norm to apply for Minkowski, weighted and unweighted.
        Default: 2.

    Returns : Pandas Data Frame having distance matrix created by function
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

    # add the columns time and animal id to the window needed for group by and
    # the column generation
    normalized_df[not_allowed_keys] = data[not_allowed_keys]

    # compute the distance for each time moment
    df2 = normalized_df.groupby('time').apply(similarity_computation, w=w, p=p)

    # combine the distance matrix with the data and return
    return pd.merge(data, df2, left_index=True, right_index=True).sort_values(by=['time', 'animal_id'])


def similarity_computation(group, w, p):

    """
    Compute the minkowski similarity for one individual grouped time step using
    the Scipy pdist and squareform methods

    # Usage example
        if __name__ == "__main__":
            path_to_file = "examples/datasets/fish-5.csv"
            # Read in CSV file using 'path_to_file' variable-
            data = movekit.io.parse_csv(path_to_file)
            data_grouped = movekit.preprocess.grouping_data(data)
            data_features = movekit.feature_extraction.compute_absolute_features(
                            data_grouped)
            # print(data_features)
            weights = {'Distance': 1,  'Average_Speed': 1,
                       'Average_Acceleration': 1, 'x': 1, 'y': 1}
            result_data = compute_similarity(data_features, weights)
            print(result_data)
    """

    # ids of each animal
    ids = group['animal_id'].tolist()

    # compute and assign the distances for each time step
    return pd.DataFrame(squareform(pdist(group, 'wminkowski', p=p, w=w)),
                        index=group.index, columns=ids)


def calculate_centroid(data_groups):

    """
    Computes the distance of each animal from the centroid of the group

    Input: Expects a dictionary containing-
    animal_id as key and Pandas Data Frame for that animal_id as value

    Use Pandas group by to create such a Python 3 dictionary

    Returns: A modified Pandas Data Frame containing 'distance_centroid'
    attribute
    """

    for group in data_groups.keys():
        x_mean = data_groups[group]['x'].mean()
        y_mean = data_groups[group]['y'].mean()
        # print("\nGroup = {0}, x_mean = {1:.3f} and y_mean = {2:.3f}".format(group, x_mean, y_mean))

        x = np.asarray(data_groups[group]['x'])
        y = np.asarray(data_groups[group]['y'])

        x_temp = (x - x_mean)**2
        y_temp = (y - y_mean)**2
        dist = np.sqrt(x_temp + y_temp)

        data_groups[group] = data_groups[group].assign(
            distance_to_centroid=np.around(dist, decimals=3))

    # Show 'distance_x' attribute less than zero-
    # data_groups[905].loc[data_groups[905]['distance_x'] < 0, ]

    # Concatenate different groups into one Pandas DataFrame-
    result = pd.concat(data_groups[aid] for aid in data_groups.keys())

    # Reset indices-
    result.reset_index(drop=True, inplace=True)

    # Write file to HDD (optional)-
    # result.to_csv("fish-5_centroid.csv", index=False)

    return result
    # return data_groups


def medoid_computation(data):

    """
    Calculates the data point (animal_id) closest to center/centroid for a time
    step. Uses group by on 'time' attribute.

    Input:      Expects a Pandas CSV input parameter containing the dataset
    Returns:    Python 3 dictionary having as key, 'time' and as values,
                Pandas DataFrame associated with it

    # Example usage:

        # Read in data-
        data = pd.read_csv("fish-5.csv")

        # Sort values by 'time' attribute-
        data.sort_values("time", ascending=True, inplace = True)

        # Group according to 'animal_id' attribute-
        data_animals = data.groupby('animal_id')

        # A dict object to store different groups created above as-
        # animal_id: pandas dataframe for that animal_id
        data_groups = {}

        # Add different 'animal_id' in dict-
        for aid in data_animals.groups.keys():
        	data_groups[aid] = data_animals.get_group(aid)

        # Reset index-
        for aid in data_animals.groups.keys():
        	data_groups[aid].reset_index(drop = True, inplace = True)

        # Compute centroid-
        data_centroid = calculate_centroid(data_groups)

        # Compute medoid-
        data_medoid = medoid_computation(data)

        # Check for rows where 'medod' != 312-
        data_medoid[data_medoid['medoid'] != 312]

        # Optional (Save to HDD)-
        data_medoid.to_csv("medoid_computation.csv", index=False)

    """

    # Group by 'time'-
    # Group according to 'animal_id' attribute-
    data_time = data.groupby('time')

    # Dictionary to hold grouped data by 'time' attribute-
    data_groups_time = {}

    for aid in data_time.groups.keys():
        data_groups_time[aid] = data_time.get_group(aid)

    # Reset index-
    for aid in data_time.groups.keys():
        data_groups_time[aid].reset_index(drop=True, inplace=True)

    # Each group has the dimension-
    # data_groups_time[10].shape
    # (5, 4)

    for tid in data_groups_time.keys():

        # Compute centroid for each time step-
        x_mean = data_groups_time[tid]['x'].mean()
        y_mean = data_groups_time[tid]['y'].mean()

        # Centroid of this group-
        # x_mean, y_mean

        # print("\nCentroid of this group: x = {0:.4f} & y: {1:.4f}\n".format(x_mean, y_mean))

        x = np.asarray(data_groups_time[tid]['x'])
        y = np.asarray(data_groups_time[tid]['y'])

        x_temp = (x - x_mean) ** 2
        y_temp = (y - y_mean) ** 2

        dist = np.sqrt(x_temp + y_temp)

        data_groups_time[tid] = data_groups_time[tid].assign(
            distance_to_centroid=np.around(dist, decimals=3))

        # Find 'animal_id' nearest to centroid for this group-
        pos = np.argmin(data_groups_time[tid]['distance_to_centroid'].values)
        nearest = data_groups_time[tid].loc[pos, 'animal_id']

        # Assign 'medoid' for this group-
        data_groups_time[tid] = data_groups_time[tid].assign(medoid=nearest)

    # Concatenate different groups into one Pandas DataFrame-
    result = pd.concat(data_groups_time[aid]
                       for aid in data_groups_time.keys())

    # Reset indices-
    result.reset_index(drop=True, inplace=True)

    # return data_groups_time
    return result
