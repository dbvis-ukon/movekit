import numpy as np
import pandas as pd
import warnings
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import tsfresh
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from geoalchemy2 import functions, elements


def grouping_data(processed_data, pick_vars=None):
    """
    Function to group data records by 'animal_id'. Adds additional attributes/columns, if features aren't extracted yet.
    :param processed_data: pd.DataFrame with all preprocessed records.
    :return: dictionary with 'animal_id' as key and all records as value.
    """

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
    if list(processed_data.columns.values) == list(
        ['time', 'animal_id', 'x', 'y']):
        # Add additional attributes/columns to each groups-
        for aid in data_animal_id_groups.keys():
            data = [None for x in range(data_animal_id_groups[aid].shape[0])]
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
            data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
                stopped=data)
    if pick_vars != None:
        for aid in data_animal_id_groups.keys():
            data_animal_id_groups[aid] = data_animal_id_groups[
                aid].loc[:, pick_vars]
    return data_animal_id_groups


def timewise_dict(data):
    """
    Group records by timestep.
    :param data: pd.DataFrame with all preprocessed records.
    :return: dictionary with 'time' as key and all animal records as value.
    """
    # Input grouped into time
    data_time = data.groupby('time')

    # Pour into dictionary to hold data by time
    dict_time = {}

    for time in data_time.groups.keys():
        dict_time[time] = data_time.get_group(time)
        dict_time[time].reset_index(drop=True, inplace=True)

    return dict_time


def regrouping_data(data_animal_id_groups):
    """
    Concatenate all Pandas DataFrames in grouped dictionary into one.
    :param data_animal_id_groups: dictionary ordered by 'animal_id'.
    :return: Pandas DataFrame containing all records of all andimal_ids.
    """
    # Concatenate all single records into one
    result = pd.concat(data_animal_id_groups[aid]
                       for aid in data_animal_id_groups.keys())

    result.sort_values(['animal_id', 'time'], ascending=True, inplace=True)
    # Reset index-
    result.reset_index(drop=True, inplace=True)
    return result


def compute_direction(data_animal_id_groups,
                      param_x="x",
                      param_y="y",
                      colname="direction"):
    """
    Calculate angle of degrees, an animal is heading in between two timesteps.
    :param data_animal_id_groups: dictionary ordered by 'animal_id'.
    :param param_x: Column name to be recognized as x. Default "x".
    :param param_y: Column name to be recognized as y. Default "y".
    :return: dictionary containing computed 'distance' attribute.
    """
    # Compute 'direction' for 'animal_id' groups-
    for aid in data_animal_id_groups.keys():
        data = np.rad2deg(
            np.arctan2((data_animal_id_groups[aid][param_y] -
                        data_animal_id_groups[aid][param_y].shift(periods=1)),
                       (data_animal_id_groups[aid][param_x] -
                        data_animal_id_groups[aid][param_x].shift(periods=1))))
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            inp=data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].rename(
            columns={'inp': colname})
    return data_animal_id_groups


def compute_distance(data_animal_id_groups, param_x="x", param_y="y"):
    """
    Calculate metric distance of animals in between two timesteps.
    :param data_animal_id_groups: dictionary ordered by 'animal_id'.
    :param param_x: Column name to be recognized as x. Default "x".
    :param param_y: Column name to be recognized as y. Default "y".
    :return: dictionary containing computed 'distance' attribute.
    """
    for aid in data_animal_id_groups.keys():
        p1 = data_animal_id_groups[aid].loc[:, [param_x, param_y]]
        p2 = data_animal_id_groups[aid].loc[:, [param_x, param_y]].shift(
            periods=1)
        p2.iloc[0, :] = [0.0, 0.0]

        data_animal_id_groups[aid]['distance'] = ((p1 -
                                                   p2)**2).sum(axis=1)**0.5

    # Reset first entry for each 'animal_id' to zero-
    for aid in data_animal_id_groups.keys():
        data_animal_id_groups[aid].loc[0, 'distance'] = 0.0

    return data_animal_id_groups


def compute_distance_and_direction(data_animal_id_groups):
    """
    Function to calculate metric distance and direction attributes.
    Calculates the metric distance between two consecutive time frames/time stamps
    for each moving entity (in this case, fish).
    :param data_animal_id_groups: dictionary ordered by 'animal_id'.
    :return: dictionary containing computed 'distance' and 'direction' attributes.
    """

    # Compute 'direction' for 'animal_id' groups-
    for aid in data_animal_id_groups.keys():
        data = []
        try:
            data = np.rad2deg(
                np.arctan2((data_animal_id_groups[aid]['y'] -
                            data_animal_id_groups[aid]['y'].shift(periods=1)),
                           (data_animal_id_groups[aid]['x'] -
                            data_animal_id_groups[aid]['x'].shift(periods=1))))
        except TypeError:
            data = 0

        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            direction=data)

    # Compute 'distance' for 'animal_id' groups-
    for aid in data_animal_id_groups.keys():
        data = []
        try:
            p1 = data_animal_id_groups[aid].loc[:, ['x', 'y']]
            p2 = data_animal_id_groups[aid].loc[:, ['x', 'y']].shift(periods=1)
            p2.iloc[0, :] = [0.0, 0.0]

            data = ((p1 - p2)**2).sum(axis=1)**0.5

        except TypeError:
            data = 0

        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(
            distance=data)

    # Reset first entry for each 'animal_id' to zero-
    for aid in data_animal_id_groups.keys():
        data_animal_id_groups[aid].loc[0, 'distance'] = 0.0

    return data_animal_id_groups


def compute_average_speed(data_animal_id_groups, fps):
    """
    Compute average speed of an animal based on fps (frames per second) parameter.
    Formula used Average Speed = Total Distance Travelled / Total Time taken;
    Use output of compute_distance_and_direction() function to this function.
    :param data_animal_id_groups: dictionary with 'animal_id' as keys
    :param fps: integer to specify frames per second
    :return: dictionary, including measure for 'average_speed'
    """
    for aid in data_animal_id_groups.keys():
        data_animal_id_groups[aid]['average_speed'] = data_animal_id_groups[
            aid]['distance'].rolling(min_periods=1, window=fps,
                                     center=True).mean().fillna(0)
    return data_animal_id_groups


def compute_average_acceleration(data_animal_id_groups, fps):
    """
    Compute average acceleration of an animal based on fps (frames per second) parameter.
    Formulas used are- Average Acceleration = (Final Speed - Initial Speed) / Total Time Taken;
    Use output of compute_average_speed() function to this function.
    :param data_animal_id_groups: dictionary with 'animal_id' as keys
    :param fps: integer to specify frames per second
    :return: dictionary, including measure for 'average_acceleration'
    """
    for aid in data_animal_id_groups.keys():

        # rename into shortcut
        speed = data_animal_id_groups[aid]['average_speed']
        #b = data_animal_id_groups[aid]['average_speed'].shift(periods=1)
        try:
            data_animal_id_groups[aid]['average_acceleration'] = speed.rolling(
                min_periods=1, window=fps,
                center=True).apply(lambda x: x[1] - x[0], raw=True).fillna(0)
        except:
            data_animal_id_groups[aid]['average_acceleration'] = 0

    return data_animal_id_groups


def extract_features(data, fps=10, stop_threshold=0.5):
    """
    Calculate and return all absolute features for input animal group.
    Combined usage of the functions on DataFrame grouping_data(), compute_distance_and_direction(), compute_average_speed(),
    compute_average_acceleration(), computing_stops()
    :param data: pandas DataFrame with all records of movements.
    :param fps: integer to specify frames per second.
    :param stop_threshold: integer to specify threshold, at which we consider a "stop".
    :return: pandas DataFrame with additional variables consisting of all relevant features.
    """
    tmp_data = grouping_data(data)
    tmp_data = compute_distance_and_direction(tmp_data)
    tmp_data = compute_average_speed(tmp_data, fps)
    tmp_data = compute_average_acceleration(tmp_data, fps)
    tmp_data = computing_stops(tmp_data, stop_threshold)

    # Regroup dictionary into pd DataFrame
    regrouped_data = regrouping_data(tmp_data)

    # Replace NA
    regrouped_data.fillna(0, inplace=True)

    return regrouped_data


def computing_stops(data_animal_id_groups, threshold_speed):
    """
    Calculate absolute feature, describing a record as stop, based on threshold.
    Calculate absolute feature called 'Stopped' where the value is 1 if 'Average_Speed' <= threshold_speed
    and 0 otherwise.
    :param data_animal_id_groups: dictionary with 'animal_id' as keys.
    :param threshold_speed:  integer, defining maximum value for 'average_speed' to be considered as a stop.
    :return: dictionary, including variable 'stopped'.
    """

    for aid in data_animal_id_groups.keys():
        data_animal_id_groups[aid]['stopped'] = np.where(
            data_animal_id_groups[aid]['average_speed'] <= threshold_speed, 1,
            0)
    return data_animal_id_groups
    '''
    data_animal_id_groups['stopped'] = np.where(
        data_animal_id_groups['average_speed'] <= threshold_speed, 1, 0)

    return data_animal_id_groups
    '''


def group_movement(feats):
    """
    Returns aggregated movement data, such as distance, mean speed, mean acceleration and mean distance to centroid for the entire group at each time capture.

    :param feats: pd DataFrame with animal-specific data - if no features contained, they will be extracted.
    :return: pd DataFrame with group-specific values for each time-capture

    """

    # Handling no features in input
    if 'distance' not in feats.columns or 'average_speed' not in feats.columns or 'average_acceleration' not in feats.columns:
        warnings.warn(
            'Recalculating features, since distance, speed or acceleration not found in input.'
        )
        feats = extract_features(feats)

    # Handling no centroid in input
    if 'distance_to_centroid' not in feats.columns:
        warnings.warn(
            'Recalculating centroid-distances, since not found in input dataset!'
        )
        feats = centroid_medoid_computation(feats)
    # Group by time, return new dataframe
    data_dist = feats.groupby('time')

    group = pd.DataFrame({
        "total_dist":
        data_dist.sum()['distance'],
        "mean_speed":
        data_dist.mean()['average_speed'],
        "mean_acceleration":
        data_dist.mean()['average_acceleration'],
        "mean_distance_centroid":
        data_dist.mean()['distance_to_centroid']
    })

    return group


def centroid_medoid_computation(data,
                                only_centroid=False,
                                object_output=False):
    """
    Calculates the data point (animal_id) closest to center/centroid/medoid for a time step
    Uses group by on 'time' attribute
    :param data: Pandas DataFrame containing movement records
    :param data: Boolean in case we just want to compute the centroids. Default: False.
    :return: Pandas DataFrame containing computed medoids & centroids
    """
    # Group by 'time'-
    data_time = data.groupby('time')

    # Dictionary to hold grouped data by 'time' attribute-
    data_groups_time = {}

    for aid in data_time.groups.keys():
        data_groups_time[aid] = data_time.get_group(aid)
        data_groups_time[aid].reset_index(drop=True, inplace=True)

        # NOTE:
        # Each group has only five entries
        # Each group has dimension- (5, 4)

        # Add 3 additional columns to each group-
        data_l = [0 for x in range(data_groups_time[aid].shape[0])]

        data_groups_time[aid] = data_groups_time[aid].assign(x_centroid=data_l)
        data_groups_time[aid] = data_groups_time[aid].assign(y_centroid=data_l)

        if only_centroid == False:
            data_groups_time[aid] = data_groups_time[aid].assign(medoid=data_l)
            data_groups_time[aid] = data_groups_time[aid].assign(
                distance_to_centroid=data_l)

        # Calculate centroid coordinates (x, y)-
        x_mean = np.around(np.mean(data_groups_time[aid]['x']), 3)
        y_mean = np.around(np.mean(data_groups_time[aid]['y']), 3)

        data_groups_time[aid] = data_groups_time[aid].assign(x_centroid=x_mean)
        data_groups_time[aid] = data_groups_time[aid].assign(y_centroid=y_mean)

        # add additional variable for point object
        if object_output == True:
            data_groups_time[aid] = data_groups_time[aid].assign(
                centroid=functions.ST_MakePoint(x_mean, y_mean))

        if only_centroid == False:
            # Squared distance of each 'x' coordinate to 'centroid'-
            x_temp = (data_groups_time[aid].loc[:, 'x'] - x_mean)**2

            # Squared distance of each 'y' coordinate to 'centroid'-
            y_temp = (data_groups_time[aid].loc[:, 'y'] - y_mean)**2

            # Distance of each point from centroid-
            dist = np.sqrt(x_temp + y_temp)

            # Assign computed distances to 'distance_to_centroid' attribute-
            data_groups_time[aid] = data_groups_time[aid].assign(
                distance_to_centroid=np.around(dist, decimals=3))

            # Find 'animal_id' nearest to centroid for this group-
            pos = np.argmin(
                data_groups_time[aid]['distance_to_centroid'].values)
            nearest = data_groups_time[aid].loc[pos, 'animal_id']

            # Assign 'medoid' for this group-
            data_groups_time[aid] = data_groups_time[aid].assign(
                medoid=nearest)

    medoid_data = regrouping_data(data_groups_time)
    return medoid_data


#DEAD below? - gives almost exact result as euclidean_dist() function.
def distance_euclidean_matrix(data):
    """
    Calculates record's euclidean distances.
    Displays euclidean distances as a distance matrix of each animal at a given point in time to each other animal at
    the same point in time, sorted by 'time' and 'animal_id'.
    :param data: pandas DataFrame, containing preprocessed movement records.
    :return: pandas DataFrame with euclidean distances to each other 'animal_id' at a given time.
    """
    return data.groupby('time').apply(euclidean_dist).sort_values(
        by=['time', 'animal_id'])


def euclidean_dist(data):
    """
    Calculate distances
    Compute the euklidean distance for one individual grouped time step using the Scipy 'pdist' and 'squareform' methods
    :param data: pandas DataFrame with positional record data.
    :return: pandas DataFrame, including computed similarities.
    """
    weights = {'x': 1, 'y': 1}
    out = compute_similarity(data, weights)
    return out


def compute_similarity(data, weights, p=2):
    """
    Compute positional similarity between animals.
    Computing the positional similarity in a distance matrix according to animal_id for each time step.
    :param data: pandas DataFrame, containing preprocessed movement records.
    :param weights: dictionary, giving variable's weights in weighted distance calculation.
    :param p: integer, giving p-norm for Minkowski, weighted and unweighted. Default: 2.
    :return: pandas DataFrame, including computed similarities.
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
    Compute similarity between records.
    Compute the Minkowski similarity for one individual grouped time step using the Scipy pdist and squareform methods
    :param group: pandas DataFrame, containing preprocessed movement records.
    :param w: array, consisting of the weight vector.
    :param p: double, applies the respective p-norm for weighted Minkowski.
    :return: pandas DataFrame, including the distances of the records.
    """
    # ids of each animal
    ids = group['animal_id'].tolist()
    # compute and assign the distances for each time step
    return pd.DataFrame(squareform(pdist(group, 'wminkowski', p=p, w=w)),
                        index=group.index,
                        columns=ids)


def ts_all_features(data):
    """
    Perform time series analysis on record data.
    Remove the column 'stopped' as it has nominal values
    :param data: pandas DataFrame, containing preprocessed movement records and features.
    :return: pandas DataFrame, containing autocorrelation for each id for each feature.
    """

    rm_colm = ['stopped']
    df = data[data.columns.difference(rm_colm)]

    time_series_features = tsfresh.extract_features(df,
                                                    column_id='animal_id',
                                                    column_sort='time')

    tsfresh.utilities.dataframe_functions.impute(time_series_features)

    return time_series_features


def ts_feature(data, feature):
    """
    Perform time series analysis on specified feature of record data.
    :param data: pandas DataFrame, containing preprocessed movement records and features.
    :param feature: feature to perform time series analysis on
    :return: pandas DataFrame, containing autocorrelation for each id for defined feature.
    """
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
    Show percentage of environment space explored by singular animal.
    Using minumum and maximum of 2-D coordinates, given by 'x' and 'y' features in input DataFrame.
    :param data: pandas DataFrame, containing preprocessed movement records.
    :return: None.
    """
    # Compute global minimum and maximum if user
    # has NOT specified the values-
    x_min = data['x'].min()
    x_max = data['x'].max()

    y_min = data['y'].min()
    y_max = data['y'].max()

    # Group 'data' using 'animal_id' attribute-
    data_groups = grouping_data(data)

    # for each animal, how much of the area has the animal covered?
    # % of space explored by each animal? #
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


def explore_features_geospatial(preprocessed_data):
    """
    Show exploration of environment space by each animal using 'shapely' package.
    Gives singular descriptions of polygon area covered by each animal and combined.
    Additionally a plot of the respective areas is provided.
    :param preprocessed_data: pandas DataFrame, containing preprocessed movement records.
    :return: None.
    """
    # Create dictionary, grouping records by animal ID as key
    data_groups = grouping_data(preprocessed_data)

    # Python dict to hold X-Y coordinates for each animal-
    xy_coord = {}

    # Polygon for singular animal ID
    for aid in data_groups.keys():
        xy_coord[aid] = []

    # Extract position information per animal into list of xy-tuples
    for aid in data_groups.keys():
        for x in range(data_groups[aid].shape[0]):
            temp_tuple = (data_groups[aid].loc[x,
                                               'x'], data_groups[aid].loc[x,
                                                                          'y'])
            xy_coord[aid].append(temp_tuple)

    # Polygons for individual animals
    for aid in data_groups.keys():
        poly = Polygon(xy_coord[aid]).convex_hull

        # Compute area of singular polygons and plot
        print(
            "\nArea (polygon) covered by animal ID = {0} is = {1:.2f} sq. units\n"
            .format(aid, poly.area))
        plt.plot(*poly.exterior.xy)

    # Polygon for collective group
    xy_coord_full = []
    for aid in data_groups.keys():
        for x in range(data_groups[aid].shape[0]):
            temp_tuple = (data_groups[aid].loc[x,
                                               'x'], data_groups[aid].loc[x,
                                                                          'y'])
            xy_coord_full.append(temp_tuple)

    # Create 'Polygon' object using all coordinates for animal ID combined
    full_poly = Polygon(xy_coord_full).convex_hull

    # Compute area of collective polygon and plot
    print("\nArea (polygon) covered by animals collectively is = ",
          full_poly.area, "sq. units")
    plt.plot(*full_poly.exterior.xy, linewidth=5, color="black")
    plt.show()
    return None


def outlier_detection(dataset, features = ["distance","average_speed", "average_acceleration","direction",
                                            "stopped"], contamination = 0.01, n_neighbors = 5, method = "mean", \
                                                                                           metric = "minkowski"):
    """
    Detect outliers based on pyod KNN.

    Note: User may decide upon contamination threshold, number of neighbors, method and metric.
    For method three kNN detectors are supported:
        -largest: use the distance to the kth neighbor as the outlier score
        -mean(default): use the average of all k neighbors as the outlier score
        -median: use the median of the distance to k neighbors as the outlier score

    :param dataset: list of features to detect outliers upon.
    :param contamination: float in (0., 0.5),  (default=0.01) The amount of contamination of the data set,
    i.e. the proportion of outliers in the data set.
    :param n_neighbors: int, (default = 5) Number of neighbors to use by default for k neighbors queries.
    :param method: str, (default='largest') {'largest', 'mean', 'median'}
    :param metric: string or callable, default 'minkowski' metric to use for distance computation. Any metric from
    scikit-learn or scipy.spatial.distance can be used.
    :return:
    """
    clf = KNN(contamination=contamination,
              n_neighbors=n_neighbors,
              method=method,
              metric=metric)
    inp_data = dataset.loc[:, features]

    clf.fit(inp_data)
    scores_pred = clf.predict(inp_data)

    # avoid overwriting input

    # Inserting column, with 1 if outlier, else 0
    if "outlier" in dataset:
        dataset["outlier"] = scores_pred
    else:
        dataset.insert(2, "outlier", scores_pred)
    return dataset
