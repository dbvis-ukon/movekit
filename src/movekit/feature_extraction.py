import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import tsfresh


def grouping_data(processed_data):
    '''
	A function to group all values for each 'animal_id'
	Input is 'processed_data' which is processed Pandas DataFrame
	Returns a dictionary where-
	key is animal_id, value in Pandas DataFrame for that 'animal_id'
	'''

    # A dictionary object to hold all groups obtained using group by-
    # Apply grouping using 'animal_id' attribute-
    data_animal_id = processed_data.groupby('animal_id')

    # A dictionary object to hold all groups obtained using group by-
    data_animal_id_groups = {}

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
        # data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
        #     positive_acceleration=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            direction=data)

    return data_animal_id_groups


def compute_distance_and_direction(data_animal_id_groups):
    '''
	Function to calculate metric distance and direction attributes
	Calculate the metric distance between two consecutive time frames/time stamps
	for each moving entity (in this case, fish)

	Use output of grouping_data() function to this function.

	Accepts a Python 3 dictionary
	Returns a Python 3 dictionary containing computed 'distance'
	and 'direction' attributes
	'''

    # Compute 'direction' for 'animal_id' groups-
    for aid in data_animal_id_groups.keys():
        data_animal_id_groups[aid]['direction'] = np.rad2deg(
            np.arctan2((data_animal_id_groups[aid]['y'] -
                        data_animal_id_groups[aid]['y'].shift(periods=1)),
                       (data_animal_id_groups[aid]['x'] -
                        data_animal_id_groups[aid]['x'].shift(periods=1))))

    # Compute 'distance' for 'animal_id' groups-
    for aid in data_animal_id_groups.keys():
        p1 = data_animal_id_groups[aid].loc[:, ['x', 'y']]
        p2 = data_animal_id_groups[aid].loc[:, ['x', 'y']].shift(periods=1)
        p2.iloc[0, :] = [0.0, 0.0]

        data_animal_id_groups[aid]['distance'] = ((p1 -
                                                   p2)**2).sum(axis=1)**0.5

    # Reset first entry for each 'animal_id' to zero-
    for aid in data_animal_id_groups.keys():
        data_animal_id_groups[aid].loc[0, 'distance'] = 0.0

    return data_animal_id_groups


def compute_average_speed(data_animal_id_groups, fps):
    '''
	Function to compute average speed of an animal based on fps
	(frames per second) parameter. Calculate the average speed of a mover,
	based on the pandas dataframe and a frames per second (fps) parameter

	Formula used-
	Average Speed = Total Distance Travelled / Total Time taken

	Use output of compute_distance_and_direction() function to this function.

	Input- Python dict and fps
	Returns- Python dict
	'''
    for aid in data_animal_id_groups.keys():
        data_animal_id_groups[aid]['average_speed'] = data_animal_id_groups[aid] \
        ['distance'].rolling(window = fps, win_type = None).sum() / fps

    return data_animal_id_groups


def compute_average_acceleration(data_animal_id_groups, fps):
    '''
	A function to compute average acceleration of an animal based on fps
	(frames per second) parameter.

	Formulas used are-
	Average Acceleration = (Final Speed - Initial Speed) / Total Time Taken

	Use output of compute_average_speed() function to this function.

	Input- Python 3 dict and fps
	Returns- Pandas DataFrame containing computations
	'''
    for aid in data_animal_id_groups.keys():
        a = data_animal_id_groups[aid]['average_speed']
        b = data_animal_id_groups[aid]['average_speed'].shift(periods=1)

        data_animal_id_groups[aid]['average_acceleration'] = (a - b) / fps

    # Concatenate all Pandas DataFrame into one-
    result = pd.concat(data_animal_id_groups[aid]
                       for aid in data_animal_id_groups.keys())

    # Reset index-
    result.reset_index(drop=True, inplace=True)

    return result


def compute_absolute_features(data, fps=10, stop_threshold=0.5):
    '''
	Calculate absolute features for the input data animal group.

	Input- Python 3 dictionary, fps (frames per second) and stopping threshold
	Returns- Pandas Python 3 dictionary
	'''

    direction_distance_data = compute_distance_and_direction(
        data_animal_id_groups)

    avg_speed_data = compute_average_speed(direction_distance_data, fps)

    avg_acceleration_data = compute_average_acceleration(avg_speed_data, fps)

    stop_data = computing_stops(avg_acceleration_data, stop_threshold)

    return stop_data


def extract_features(data, fps=10, stop_threshold=0.5):
    """
	Calculate absolute features for the input data animal group.

	Input- Python 3 dictionary, fps (frames per second) and stopping threshold
	Returns- Pandas Python 3 dictionary
	"""
    tmp_data = grouping_data(data)

    tmp_data = compute_distance_and_direction(tmp_data)

    tmp_data = compute_average_speed(tmp_data, fps)

    tmp_data = compute_average_acceleration(tmp_data, fps)

    tmp_data = computing_stops(tmp_data, stop_threshold)

    tmp_data.fillna(0, inplace=True)

    return tmp_data


def computing_stops(data_animal_id_groups, threshold_speed):
    '''
    Calculate absolute feature called 'Stopped' where the value is 'yes'
    if 'Average_Speed' <= threshold_speed and 'no' otherwise

    Input- Python 3 dictionary and threshold speed
	Returns- Python 3 dictionary
    '''
    data_animal_id_groups['stopped'] = np.where(
        data_animal_id_groups['average_speed'] <= threshold_speed, 1, 0)

    return data_animal_id_groups


def medoid_computation(data):
    '''
	Calculates the data point (animal_id) closest to
	center/centroid/medoid for a time step
    Uses group by on 'time' attribute

    Input-      Expects a Pandas CSV input parameter containing the dataset
    Returns-    Pandas DataFrame containing computed medoids & centroids
    '''

    # Create Python dictionary to hold final medoid computation-
    data_d = {
        'time': [0 for x in range(data.shape[0])],
        'x_coordinate_centroid': [0 for x in range(data.shape[0])],
        'y_coordinate_centroid': [0 for x in range(data.shape[0])],
        'medoid': [0 for x in range(data.shape[0])]
    }

    # Create Pandas Dataframe using dict from above-
    medoid_data = pd.DataFrame(data_d)

    # Group by 'time'-
    data_time = data.groupby('time')

    # Dictionary to hold grouped data by 'time' attribute-
    data_groups_time = {}

    for aid in data_time.groups.keys():
        data_groups_time[aid] = data_time.get_group(aid)

    # Reset index-
    for aid in data_time.groups.keys():
        data_groups_time[aid].reset_index(drop=True, inplace=True)

    # NOTE:
    # Each group has only five entries
    # Each group has dimension- (5, 4)

    # Add 3 additional columns to each group-
    for aid in data_groups_time.keys():
        data_l = [0 for x in range(data_groups_time[aid].shape[0])]

        data_groups_time[aid] = data_groups_time[aid].assign(x_centroid=data_l)
        data_groups_time[aid] = data_groups_time[aid].assign(y_centroid=data_l)
        data_groups_time[aid] = data_groups_time[aid].assign(medoid=data_l)
        data_groups_time[aid] = data_groups_time[aid].assign(
            distance_to_centroid=data_l)

    for tid in data_groups_time.keys():
        # Calculate centroid coordinates (x, y)-
        x_mean = np.around(np.mean(data_groups_time[tid]['x']), 3)
        y_mean = np.around(np.mean(data_groups_time[tid]['y']), 3)
        centroid = np.asarray([x_mean, y_mean])

        data_groups_time[tid] = data_groups_time[tid].assign(x_centroid=x_mean)
        data_groups_time[tid] = data_groups_time[tid].assign(y_centroid=y_mean)

        # Squared distance of each 'x' coordinate to 'centroid'-
        x_temp = (data_groups_time[tid].loc[:, 'x'] - x_mean)**2

        # Squared distance of each 'y' coordinate to 'centroid'-
        y_temp = (data_groups_time[tid].loc[:, 'y'] - y_mean)**2

        # Distance of each point from centroid-
        dist = np.sqrt(x_temp + y_temp)

        # Assign computed distances to 'distance_to_centroid' attribute-
        data_groups_time[tid] = data_groups_time[tid].assign(
            distance_to_centroid=np.around(dist, decimals=3))

        # Find 'animal_id' nearest to centroid for this group-
        pos = np.argmin(data_groups_time[tid]['distance_to_centroid'].values)
        nearest = data_groups_time[tid].loc[pos, 'animal_id']

        # Assign 'medoid' for this group-
        data_groups_time[tid] = data_groups_time[tid].assign(medoid=nearest)

        medoid_data.loc[tid, 'time'] = tid
        medoid_data.loc[tid, 'x_coordinate_centroid'] = x_mean
        medoid_data.loc[tid, 'y_coordinate_centroid'] = y_mean
        medoid_data.loc[tid, 'medoid'] = nearest

        # Drop index 0-
        medoid_data.drop(medoid_data.index[0], inplace=True)

    # return medoid_data, data_groups_time
    return medoid_data


def distance_euclidean_matrix(data):
    """
    A function to create a distance matrix according to animal_id for each
    time step

    Input: Pandas Data Frame containing CSV file
    Output: Pandas Data Frame having distance matrix created by function

    example usage
    distance_matrix = distance_euclidean_matrix(data)
    """
    return data.groupby('time').apply(euclidean_dist).sort_values(
        by=['time', 'animal_id'])


def euclidean_dist(data):
    """
    Compute the distance for one individual grouped time step using the
    Scipy pdist and squareform methods
    """
    weights = {'x': 1, 'y': 1}
    return compute_similarity(data, weights)


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
    normalized_df = (df - df.min()) / (df.max() - df.min())
    # add the columns time and animal id to the window needed for group by and the column generation
    normalized_df[not_allowed_keys] = data[not_allowed_keys]
    # compute the distance for each time moment
    df2 = normalized_df.groupby('time').apply(similarity_computation, w=w, p=p)
    # combine the distance matrix with the data and return
    return pd.merge(data, df2, left_index=True,
                    right_index=True).sort_values(by=['time', 'animal_id'])


def similarity_computation(group, w, p):
    """
    Compute the minkowski similarity for one individual grouped time step using the Scipy pdist and squareform methods
    """
    # ids of each animal
    ids = group['animal_id'].tolist()
    # compute and assign the distances for each time step
    return pd.DataFrame(squareform(pdist(group, 'wminkowski', p=p, w=w)),
                        index=group.index,
                        columns=ids)


def ts_all_features(data):
    '''
	Function to perform time series analysis on provided
	dataset.
	Remove the columns stopped as it has nominal values
	'''

    rm_colm = ['stopped']
    df = data[data.columns.difference(rm_colm)]

    time_series_features = tsfresh.extract_features(df,
                                                    column_id='animal_id',
                                                    column_sort='time')

    tsfresh.utilities.dataframe_functions.impute(time_series_features)

    return (time_series_features)


def ts_feature(data, feature):
    '''
	Function to perform time series analysis on provided
	dataset with the specific feature.
	Remove the columns stopped as it has nominal values

    Input:
	data 	-	Pandas DataFrame (should be sorted by 'time' attribute)
    feature     String feature which defines which feature should be extracted
	'''
    fc_parameters = tsfresh.feature_extraction.ComprehensiveFCParameters()
    if feature in fc_parameters:
        settings = {}
        settings[feature] = fc_parameters[feature]

        rm_colm = ['stopped']
        df = data[data.columns.difference(rm_colm)]
        time_series_features = tsfresh.extract_features(
            df,
            column_id='animal_id',
            column_sort='time',
            default_fc_parameters=settings)
        return time_series_features
    else:
        print("Time series feature is not known.")
        return


def explore_features(data):
    """
	Function to perform percentage of environment space
	explored by each animal using minumum and maximum of
	2-D coordinates

	Input:
	data-		Pandas DataFrame containing data

	Returns:	None
	"""
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    # Compute global minimum and maximum if user
    # has NOT specified the values-
    x_min = data['x'].min()
    x_max = data['x'].max()

    y_min = data['y'].min()
    y_max = data['y'].max()

    # Group 'data' using 'animal_id' attribute-
    data_groups = grouping_data(data)

    # for each animal, how much of the area has the animal covered?
    # % of space explored by each animal?
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

    return None


# def explore_features_geospatial(data_groups):
#     """
# 	Function to perform exploration of environment space
# 	by each animal using 'shapely' package

# 	Input:
# 	data_groups-	Python 3 dictonary containing
# 					grouping of data by 'animal_id'
# 					attribute

# 	Returns:	None
# 	"""

#     # Python dict to hold X-Y coordinates for each animal-
#     xy_coord = {}

#     for aid in data_groups.keys():
#         xy_coord[aid] = []

#     for aid in data_groups.keys():
#         for x in range(data_groups[aid].shape[0]):
#             temp_tuple = (data_groups[aid].loc[x, 'x'],
#                           data_groups[aid].loc[x, 'y'])
#             xy_coord[aid].append(temp_tuple)

#     for aid in data_groups.keys():
#         # Creat a 'Polygon' object using all coordinates for animal ID-
#         poly = Polygon(xy_coord[aid])

#         # Compute area of polygon-
#         print(
#             "\nArea (polygon) covered by animal ID = {0} is = {1:.2f} sq. units\n"
#             .format(aid, poly.area))

#         # OPTIONAL:
#         # Plot shapely polygon and objects-
#         # plt.plot(*poly.exterior.xy)
#         # plt.show()

#     return None
