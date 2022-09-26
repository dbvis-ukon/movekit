import numpy as np
import pandas as pd
import warnings
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import tsfresh
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from .utils import presence_3d
from functools import reduce
import st_clustering as stc
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from pandas.api.types import is_numeric_dtype

from .feature_extraction import *
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, convex_hull_plot_2d, Delaunay, delaunay_plot_2d


def get_trajectories(data_groups):
    """
    Obtain trajectories out of a grouped dictionary with multiple ids.
    :param data_groups: Grouped dictionary by animal_id.
    :return: Grouped dictionary by animal id, containing tuples of positions in 2d coordinate system.
    """

    # create new dictionary
    trajectories = {}
    for aid in data_groups.keys():
        # add dict item, holding x-y tuples for the trajectories of each animal id
        trajectories[aid] = list(
            zip(data_groups[aid]["x"], data_groups[aid]["y"]))
    return trajectories


def dtw_matrix(preprocessed_data, path=False, distance=euclidean):
    """
    Obtain dynamic time warping amongst all trajectories from the grouped animal-records.
    :param preprocessed_data: pandas Dataframe containing the movement records.
    :param path: Boolean to specify if matrix of dtw-path gets returned as well.
    :param distance: Specify which distance measure to use. Default: "euclidean". (ex. Alternatives: pdist, minkowski)
    :return: pandas Dataframe with distances between trajectories.
    """
    data_groups = grouping_data(preprocessed_data)

    # get trajectory-dictionary with local function
    trajectories = get_trajectories(data_groups)

    # create empty np array with size, depending on number of tracked animals
    distance_matr = np.empty(
        (len([*trajectories.keys()]), len([*trajectories.keys()])))

    # create empty np list-array for paths with size, depending on number of tracked animals
    path_matr = np.empty(
        (len([*trajectories.keys()]), len([*trajectories.keys()])), dtype=list)

    # double-iterate over obtained trajectory dict
    for aid in tqdm(range(len([*trajectories.keys()])),position=0, desc="Calculating dynamic time warping"):
        for aid2 in range(len([*trajectories.keys()])):
            # fill np array field with euclidean distance of respective trajectories, same for path field
            distance_matr[aid][aid2], path_matr[aid][aid2] = fastdtw(
                trajectories[[*trajectories.keys()][aid]],
                trajectories[[*trajectories.keys()][aid2]],
                dist=distance)
            # generate pandas df from distance array
            distance_df = pd.DataFrame(data=distance_matr,
                                       index=[*trajectories.keys()],
                                       columns=[*trajectories.keys()])
    if path:
        return distance_df, path_matr
    else:
        return distance_df

def compute_centroid_direction(data, colname="centroid_direction", group_output=False, only_centroid=True):
    """
    Calculate the direction of the centroid. Calculates centroid, if not in input data.
    :param pd DataFrame: DataFrame with x/y positional data and animal_ids, optionally include centroid
    :param colname: Name of the column. Default: centroid_direction.
    :param group_output: Boolean, defines form of output. Default: Animal-Level.
    :param only_centroid: Boolean in case we just want to compute the centroids. Default: True.
    :return: pandas DataFrame with centroid direction included
    """
    # Handle centroid not in data
    if "x_centroid" not in data.columns or "y_centroid" not in data.columns:
        warnings.warn(
            'x_centroid or y_centroid not found in data. Calculating centroid...'
        )
        data = centroid_medoid_computation(data, only_centroid=only_centroid)

    # Group into animals
    dat = grouping_data(data)

    with tqdm(total=100, position=0, desc="Computing centroid direction") as pbar:
        pbar.update(10)  # because compute_direction starts at 10% due to its call in extract_features
        dat = compute_direction(dat,
                                pbar,
                                param_x="x_centroid",
                                param_y="y_centroid",
                                colname=colname)

    cen_direction = regrouping_data(dat)

    if group_output == False:
        return cen_direction

    else:
        pol = cen_direction
        return pol.loc[pol.animal_id == list(set(pol.animal_id))[0],
                       ['time', colname]].reset_index(drop=True)


def get_heading_difference(preprocessed_data):
    """
    Calculate the difference in between the animal's direction and the centroid's direction for each timestep.
    The difference is measured by the cosine similarity of the two direction vectors. The value range is from -1 to 1,
    with 1 meaning animal and centroid having the same direction while -1 meaning they have opposite directions.
    :param preprocessed_data: Pandas Dataframe containing preprocessed animal records.
    :return: Pandas Dataframe containing animal and centroid directions as well as the heading difference.
    """
    if "direction" not in preprocessed_data.columns:
        preprocessed_data = extract_features(preprocessed_data)

    if "x_centroid" not in preprocessed_data.columns or "y_centroid" not in preprocessed_data.columns:
        preprocessed_data = centroid_medoid_computation(preprocessed_data)
    # Obtain the centroid positions for each timestep, group into dictionary

    animal_dir = grouping_data(preprocessed_data)

    # Get the directions  for each centroid for each timestep
    with tqdm(total=100,position=0, desc="Calculating heading difference") as pbar:
        pbar.update(10)  # because the method compute_direction() assumes 10% are already filled
        cen_dir = compute_direction(animal_dir,
                                    pbar,
                                    param_x="x_centroid",
                                    param_y="y_centroid",
                                    colname="centroid_direction")

        directions = regrouping_data(cen_dir)
        # calculate cosine similarity of the centroids and the animals direction vector
        cos_similarities = [cosine_similarity(np.array([directions['direction'][i]]), np.array([directions['centroid_direction'][i]]))[0][0] for i in range(0, len(directions[\
        'direction']))]  # cosine similarity for direction vectors of animal and centroid
        directions['heading_difference'] = cos_similarities
    return directions


def compute_polarization(preprocessed_data, group_output=False):
    """
    Compute the polarization of a group at all record timepoints.
    More info about the formula: Here: https://bit.ly/2xZ8uSI and Here: https://bit.ly/3aWfbDv. As the formula only takes angles as input,
    the polarization is calculated for 2d - Data by first calculating the direction angles of the different movers and afterwards by calculating the polarization.
    For 3-dimensional data for all two's-combinations of the three dimensions the polarization is calculated in the way described before for 2d-data,
    afterwards the mean of the three results is taken as result for the polarization.
    :param preprocessed_data: Pandas Dataframe with or without previously extracted features.
    :return: Pandas Dataframe, with extracted features along with a new "polarization" variable.
    """

    def polarization(preprocessed_data, group_output):
        # convert to radians for polarization formula
        preprocessed_data['direction_angle'] = preprocessed_data['direction_angle'].apply(lambda x: math.radians(x))

        # Group by 'time'-
        data_time = preprocessed_data.groupby('time')

        # Dictionary to hold grouped data by 'time' attribute-
        data_groups_time = {}

        # Obtain polarization for each point in time
        for aid in data_time.groups.keys():
            data_groups_time[aid] = data_time.get_group(aid)
            data_groups_time[aid].reset_index(drop=True, inplace=True)
            data = (1 / len(data_groups_time[aid]["direction_angle"])) * np.sqrt(
                (sum(np.sin(data_groups_time[aid]["direction_angle"].astype(np.float64)))
                 )**2 +
                (sum(np.cos(data_groups_time[aid]["direction_angle"].astype(np.float64)))
                 )**2)

            data_groups_time[aid] = data_groups_time[aid].assign(polarization=data)

            # Regroup data into DataFrame
        polarization_data = regrouping_data(data_groups_time)

        # convert direction angle back to degrees
        polarization_data['direction_angle'] = polarization_data['direction_angle'].apply(lambda x: math.degrees(x))

        # If interested in fullstack output for each animal
        if group_output == False:
            return polarization_data

        # If only interested in group level output, return one line per timeslot
        else:
            pol = polarization_data
            return pol.loc[pol.animal_id == list(set(pol.animal_id))[0],
                           ['time', 'polarization']].reset_index(drop=True)

    # Check if 3d
    if 'z' in preprocessed_data.columns:
        # if 3d calculate direction angle for all three two's-combinations of the three dimensions
        preprocessed_data = preprocessed_data.rename(columns={'z': 'zz'})
        preprocessed_data_1 = compute_direction_angle(preprocessed_data)
        preprocessed_data_2 = compute_direction_angle(preprocessed_data, param_x='x', param_y='zz')
        preprocessed_data_3 = compute_direction_angle(preprocessed_data, param_x='y', param_y='zz')
        polarizations = []
        # then calculate the polarization for each combination and take the mean as the final result
        for i in [preprocessed_data_1, preprocessed_data_2, preprocessed_data_3]:
            polarizations.append(polarization(i, group_output=group_output))
        data = [(polarizations[0]['polarization'][i] + polarizations[1]['polarization'][i] + polarizations[2]['polarization'][i])
                / 3 for i in range(len(polarizations[0]['polarization']))]
        polarization_data = polarizations[0]
        polarization_data = polarization_data.assign(polarization=data)
        return polarization_data

    # if data is 2d check if it  already has direction angle calculated and afterwards calculate polarization
    else:
        if "direction_angle" not in preprocessed_data.columns:
            warnings.warn('calculating direction angle for first two dimensions, since not found in input!')
            preprocessed_data = compute_direction_angle(preprocessed_data)
            return polarization(preprocessed_data, group_output)
        else:
            return polarization(preprocessed_data, group_output)



def voronoi_volumes(points):
    """
    Function to calculate area in a voronoi-diagram. Used in function below.
    :param points: Nested list, indicating points with coordinates.
    :return: Volume for each point, infinite if area is not closed to each direction (usually outmost points).
    """
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices:  # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol


def get_spatial_objects(preprocessed_data, group_output=False):
    """
    Function to calculate convex hull, voronoi diagram and delaunay triangulation objects and also volumes of the first two objects.
    Please visit https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/spatial.html for detailed documentation of spatial attributes.
    :param preprocessed_data: Pandas Df, containing x and y coordinates.
    :param group_output: Boolean, default: False, If true, one line per time capture for entire animal group.
    :return: DataFrame either for each animal or for group at each time, containing convex hull and voronoi diagram area as well as convex hull, voronoi diagram and delaunay triangulation object.
    """

    data_time = preprocessed_data.groupby('time')

    # Dictionary to hold grouped data by 'time' attribute-
    data_groups_time = {}

    for aid in tqdm(data_time.groups.keys(),position=0, desc="Calculating spatial objects"):
        data_groups_time[aid] = data_time.get_group(aid)
        data_groups_time[aid].reset_index(drop=True, inplace=True)

        if len(data_groups_time[aid]) >= 3:  # spatial objects need minimum 3 points in timestamp
            # Obtain shape objects
            conv_hull_obj = ConvexHull(data_groups_time[aid].loc[:, ["x", "y"]])
            voronoi_obj = Voronoi(data_groups_time[aid].loc[:, ["x", "y"]])
            delaunay_obj = Delaunay(data_groups_time[aid].loc[:, ["x", "y"]])

            # Calculate area based on objects right above
            conv_hull_vol = conv_hull_obj.volume
            voronoi_vol = voronoi_volumes(data_groups_time[aid].loc[:, ["x", "y"]])

            # Assign shapes to dataframe
            data_groups_time[aid] = data_groups_time[aid].assign(
                convex_hull_object=conv_hull_obj,
                voronoi_object=voronoi_obj,
                delaunay_object=delaunay_obj)

            data_groups_time[aid] = data_groups_time[aid].assign(
                convex_hull_volume=conv_hull_vol,
                voronoi_volume=voronoi_vol,
            )

    # Regroup data into DataFrame
    out_data = regrouping_data(data_groups_time)

    if group_output == False:
        return out_data

    else:
        pol = out_data
        pol = pol.loc[pol.animal_id ==
                      list(set(pol.animal_id))[0], :].reset_index(drop=True)
        return pol.loc[:,['time', 'convex_hull_object', 'voronoi_object','delaunay_object', 'convex_hull_volume','voronoi_volume']]

def get_group_data(preprocessed_data):
    """
    Helper function to get all group data at one place.
    :param preprocessed_data: pandas DataFrame, containing preprocessed movement records.
    :return: pd DataFrame containing all relevant group variables
    """
    movement = centroid_medoid_computation(preprocessed_data)
    # prepare for merge
    movement = movement.rename(
        columns={'distance_to_centroid': 'distance_centroid'})

    # Take subset from dataset above, focusing only on group-level
    group = movement.loc[
        movement.animal_id == list(set(movement.animal_id))[0],
        ['time', 'x_centroid', 'y_centroid', 'medoid']].reset_index(drop=True)

    # compute polarization
    pol = compute_polarization(preprocessed_data, group_output=True).fillna(0)

    # compute mean speed, acceleration and mean distance to centroid
    mov = group_movement(movement).fillna(0)

    # compute centroid direction
    cen_dir = compute_centroid_direction(movement, group_output=True).fillna(0)

    # merge computed values into group-dataframe
    data_frames = [group, pol, mov, cen_dir]
    group = reduce(
        lambda left, right: pd.merge(left, right, on=['time'], how='left'),
        data_frames)
    return group


def clustering(algorithm, data, **kwargs):
    """
    Clustering of spatio-temporal data.
    :param algorithm: Choose between dbscan, hdbscan, agglomerative, kmeans, optics, spectral, affinitypropagation, birch.
    :param data: DataFrame to perform clustering on.
    :return: labels as numpy array where the label in the first position corresponds to the first row of the input data.
    """
    if algorithm == 'dbscan':
        clusterer = stc.ST_DBSCAN(**kwargs)
    elif algorithm == 'hdbscan':
        clusterer = stc.ST_HDBSCAN(**kwargs)
    elif algorithm == 'agglomerative':
        clusterer = stc.ST_Agglomerative(**kwargs)
    #elif algorithm == 'kmeans':
    #    clusterer = stc.ST_KMeans(**kwargs)
    elif algorithm == 'optics':
        clusterer = stc.ST_OPTICS(**kwargs)
    elif algorithm == 'spectral':
        clusterer = stc.ST_SpectralClustering(**kwargs)
    elif algorithm == 'affinitypropagation':
        clusterer = stc.ST_AffinityPropagation(**kwargs)   
    #elif algorithm == 'birch':
    #    clusterer = stc.ST_BIRCH(**kwargs)
    else:
        raise ValueError('Unknown algorithm. Choose between dbscan, hdbscan, agglomerative, optics, spectral, affinitypropagation.')

    if not is_numeric_dtype(data['time'][0]):  # if time format not integer
        grouped_data = data.groupby('time')
        keys = []
        for key in grouped_data.groups.keys():
            keys.append(key)
        time_distance = keys[1] - keys[0]

        for i in range(1, len(keys)):  # check if time is equidistant
            if keys[i] - keys[i - 1] != time_distance:
                warnings.warn('As difference between timestamps is not equidistant, clustering of this data is not supported by movekit at the moment.')
                return

        # convert time to integer
        time_values = np.array(data['time'])
        time_values = np.unique(time_values)
        indices = np.sort(time_values)
        converter = {}
        for i in range(len(time_values)):
            converter[indices[i]] = i
        data = data.replace({'time': converter})

    if presence_3d(data):
        data = data.loc[:, ['time','x','y','z']].values
    else:
        data = data.loc[:, ['time','x','y']].values

    clusterer.st_fit(data)
    return clusterer.labels
    
    

def clustering_with_splits(algorithm, data, frame_size, **kwargs):
    """
    Clustering of spatio-temporal data.
    :param algorithm: Choose between dbscan, hdbscan, agglomerative, optics, spectral, affinitypropagation.
    :param data: DataFrame to perform clustering on.
    :param frame_size: the dataset is partitioned into frames and merged afterwards.
    :return: labels as numpy array where the label in the first position corresponds to the first row of the input data.
    """
    if algorithm == 'dbscan':
        clusterer = stc.ST_DBSCAN(**kwargs)
    elif algorithm == 'hdbscan':
        clusterer = stc.ST_HDBSCAN(**kwargs)
    elif algorithm == 'agglomerative':
        clusterer = stc.ST_Agglomerative(**kwargs)
    #elif algorithm == 'kmeans':
    #    clusterer = stc.ST_KMeans(**kwargs)
    elif algorithm == 'optics':
        clusterer = stc.ST_OPTICS(**kwargs)
    elif algorithm == 'spectral':
        clusterer = stc.ST_SpectralClustering(**kwargs)
    elif algorithm == 'affinitypropagation':
        clusterer = stc.ST_AffinityPropagation(**kwargs)   
    #elif algorithm == 'birch':
    #    clusterer = stc.ST_BIRCH(**kwargs)
    else:
        raise ValueError('Unknown algorithm. Choose between dbscan, hdbscan, agglomerative, kmeans, optics, spectral, affinitypropagation, birch.')

    if not is_numeric_dtype(data['time'][0]):  # if time format not integer
        grouped_data = data.groupby('time')
        keys = []
        for key in grouped_data.groups.keys():
            keys.append(key)
        time_distance = keys[1] - keys[0]

        for i in range(1, len(keys)):  # check if time is equidistant
            if keys[i] - keys[i - 1] != time_distance:
                warnings.warn(
                    'As difference between timestamps is not equidistant, clustering of this data is not supported by movekit at the moment.')
                return

        # convert time to integer
        time_values = np.array(data['time'])
        time_values = np.unique(time_values)
        indices = np.sort(time_values)
        converter = {}
        for i in range(len(time_values)):
            converter[indices[i]] = i
        data = data.replace({'time': converter})

    if presence_3d(data):
        data = data.loc[:, ['time','x','y','z']].values
    else:
        data = data.loc[:, ['time','x','y']].values
    clusterer.st_fit_frame_split(data, frame_size)  # percentage bar not possible
    return clusterer.labels
