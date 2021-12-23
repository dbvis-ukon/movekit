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
    :param data_groups: Grouped dictionary by animal_id.
    :param path: Boolean to specify if matrix of dtw-path gets returned as well
    :param distance: Specify with distance measure to use. Default: "euclidean". (ex. Alternatives: pdist, minkowski)
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
    for aid in range(len([*trajectories.keys()])):
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

def compute_centroid_direction(data,
                               colname="centroid_direction",
                               group_output=False,
                               only_centroid=True):
    """Calculate the direction of the centroid. Calculates centroid, if not in input data.

    :param pd DataFrame: DataFrame with x/y positional data and animal_ids, optionally include centroid
    :param colname: Name of the column. Default: centroid_direction.
    :param group_output: Boolean, defines form of output. Default: Animal-Level
    :return: pandas DF with centroid direction included

    """
    # Handle centroid not in data
    if "x_centroid" not in data.columns or "y_centroid" not in data.columns:
        warnings.warn(
            'x_centroid or y_centroid not found in data. Calculating centroid...'
        )
        data = centroid_medoid_computation(data, only_centroid=only_centroid)

    # Group into animals
    dat = grouping_data(data)

    dat = compute_direction(dat,
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
    Calculate the difference in degrees between the animal's direction and the centroid's direction for each timestep.

    Note: Calculate the difference in degrees between the animal's direction and the centroid's direction for each
    timestep. Stronger gain in y gives positive difference, weaker gain in y gives negative difference, since constant
    y is defined to be 0 degrees.

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
    cen_dir = compute_direction(animal_dir,
                                param_x="x_centroid",
                                param_y="y_centroid",
                                colname="centroid_direction")

    # Subtract animal's direction from centroid's direction
    directions = regrouping_data(cen_dir)
    raw_diff = directions.loc[:,
                              "direction"] - directions.loc[:,
                                                            "centroid_direction"]

    # Calculate signed angle, store in new variable
    directions = directions.assign(heading_difference=(raw_diff + 180) % 360 -
                                   180)
    return directions


def compute_polarization(preprocessed_data, group_output=False):
    """
    Compute the polarization of a group at all record timepoints.

    More info about the formula: Here: https://bit.ly/2xZ8uSI and Here: https://bit.ly/3aWfbDv.
    :param preprocessed_data: Pandas Dataframe with or without previously extracted features.
    :return: Pandas Dataframe, with extracted features along with a new "polarization" variable.
    """
    # Extract features if not done yet
    if "direction" not in preprocessed_data.columns:
        warnings.warn('calculating direction, since not found in input!')
        preprocessed_data = extract_features(preprocessed_data)

    # Group by 'time'-
    data_time = preprocessed_data.groupby('time')

    # Dictionary to hold grouped data by 'time' attribute-
    data_groups_time = {}

    # Obtain polarization for each point in time
    for aid in data_time.groups.keys():
        data_groups_time[aid] = data_time.get_group(aid)
        data_groups_time[aid].reset_index(drop=True, inplace=True)
        data = (1 / len(data_groups_time[aid]["direction"])) * np.sqrt(
            (sum(np.sin(data_groups_time[aid]["direction"].astype(np.float64)))
             )**2 +
            (sum(np.cos(data_groups_time[aid]["direction"].astype(np.float64)))
             )**2)

        # Assign polarization to new variable
        data_groups_time[aid] = data_groups_time[aid].assign(polarization=data)

    # Regroup data into DataFrame
    polarization_data = regrouping_data(data_groups_time)

    # If interested in fullstack output for each animal
    if group_output == False:
        return polarization_data

    # If only interested in group level output, return one line per timeslot
    else:
        pol = polarization_data
        return pol.loc[pol.animal_id == list(set(pol.animal_id))[0],
                       ['time', 'polarization']].reset_index(drop=True)


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
    Function to calculate convex hull object, voronoi diagram and delaunay triangulation in one if no group output specified, we also obtain volumes of the first two objects.
    Please visit https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/spatial.html for detailed documentation of spatial attributes.

    :param preprocessed_data: Pandas Df, containing x and y coordinates.
    :param group_output: Boolean, default: False, If true, one line per time capture for entire animal group.
    :return: DataFrame either for each animal or for group at each time, containing convex hull area as well as convex hull object.
    """

    data_time = preprocessed_data.groupby('time')

    # Dictionary to hold grouped data by 'time' attribute-
    data_groups_time = {}

    for aid in data_time.groups.keys():
        data_groups_time[aid] = data_time.get_group(aid)
        data_groups_time[aid].reset_index(drop=True, inplace=True)

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
        return pol.drop(columns=['animal_id', 'x', 'y'])


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
    Clustering of spatio-temporal data
    :param algorithm: Choose between dbscan, hdbscan, agglomerative, kmeans, optics, spectral, affinitypropagation, birch
    :param data: DataFrame to perform clustering on
    :return: labels as numpy array where the label in the first position corresponds to the first row of the input data
    """
    if algorithm == 'dbscan':
        clusterer = stc.ST_DBSCAN(**kwargs)
    elif algorithm == 'hdbscan':
        clusterer = stc.ST_HDBSCAN(**kwargs)
    elif algorithm == 'agglomerative':
        clusterer = stc.ST_Agglomerative(**kwargs)
    elif algorithm == 'kmeans':
        clusterer = stc.ST_KMeans(**kwargs)
    elif algorithm == 'optics':
        clusterer = stc.ST_OPTICS(**kwargs)
    elif algorithm == 'spectral':
        clusterer = stc.ST_SpectralClustering(**kwargs)
    elif algorithm == 'affinitypropagation':
        clusterer = stc.ST_AffinityPropagation(**kwargs)   
    elif algorithm == 'birch':
        clusterer = stc.ST_BIRCH(**kwargs)   
    else:
        raise ValueError('Unknown algorithm. Choose between dbscan, hdbscan, agglomerative, kmeans, optics, spectral, affinitypropagation, birch.')

    if presence_3d(data):
        data = data.loc[:, ['time','x','y','z']].values
    else:
        data = data.loc[:, ['time','x','y']].values
    clusterer.st_fit(data)
    return clusterer.labels
    
    

def clustering_with_splits(algorithm, data, frame_size, **kwargs):
    """
    Clustering of spatio-temporal data
    :param algorithm: Choose between dbscan, hdbscan, agglomerative, kmeans, optics, spectral, affinitypropagation, birch
    :param data: DataFrame to perform clustering on
    :param frame_size: the dataset is partitioned into frames and merged aferwards
    :return: labels as numpy array where the label in the first position corresponds to the first row of the input data
    """
    if algorithm == 'dbscan':
        clusterer = stc.ST_DBSCAN(**kwargs)
    elif algorithm == 'hdbscan':
        clusterer = stc.ST_HDBSCAN(**kwargs)
    elif algorithm == 'agglomerative':
        clusterer = stc.ST_Agglomerative(**kwargs)
    elif algorithm == 'kmeans':
        clusterer = stc.ST_KMeans(**kwargs)
    elif algorithm == 'optics':
        clusterer = stc.ST_OPTICS(**kwargs)
    elif algorithm == 'spectral':
        clusterer = stc.ST_SpectralClustering(**kwargs)
    elif algorithm == 'affinitypropagation':
        clusterer = stc.ST_AffinityPropagation(**kwargs)   
    elif algorithm == 'birch':
        clusterer = stc.ST_BIRCH(**kwargs)   
    else:
        raise ValueError('Unknown algorithm. Choose between dbscan, hdbscan, agglomerative, kmeans, optics, spectral, affinitypropagation, birch.')

    if presence_3d(data):
        data = data.loc[:, ['time','x','y','z']].values
    else:
        data = data.loc[:, ['time','x','y']].values
    clusterer.st_fit_frame_split(data, frame_size)
    return clusterer.labels