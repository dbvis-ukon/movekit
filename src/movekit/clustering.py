import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import tsfresh
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tslearn.clustering import TimeSeriesKMeans

from .feature_extraction import *
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull


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


def ts_cluster(feats,
               n_clust,
               varlst=[
                   "distance", "average_speed", "average_acceleration",
                   "direction", "stopped"
               ],
               metric="euclidean",
               max_iter=5,
               random_state=0,
               inertia=False):
    """
    Incorporate time series clustering for absolute features.

    Note: Function can be used with extracted features beforehand. If features are not extracted, function performs
    standard feature extraction first.

    :param feats: DataFrame, containing computed features of animal record data.
    :param n_clust: Number of clusters to distinguish.
    :param varlst: List of variables to use for clustering. Default: Only 2d positions x and y.
    :param metric: Distance metric. Default: Euclidean. Alternatives: “dtw”, “softdtw”.
    :param max_iter: Max number of iterations. Default: 5.
    :param random_state: Default: 0.
    :param inertia: Additionaly return sum of distances of samples to their closest cluster center.
    :return: Default: features-dataframe with cluster and centroid columns added. Optional: inertia (see above).
    """

    # check for each feature if it's contained in feats dataset - if not, calculate
    for feat in varlst:
        if feat not in feats.columns:
            feats = extract_features(feats)
            break

    # Group data into animal-id dictionary
    data_groups = grouping_data(feats)

    # Group variables of interest for clustering into animal-id dictionary
    traj = grouping_data(feats, pick_vars=varlst)

    # Convert variables of interest to nested list
    tracks = []
    for i in [*traj.keys()]:
        tracks.append(traj[i].values.tolist())

    # Calculate timeseries k-Means based on specified parameters
    km = TimeSeriesKMeans(n_clusters=n_clust,
                          metric=metric,
                          max_iter=max_iter,
                          random_state=random_state)
    km = km.fit(tracks)
    clustcens = km.cluster_centers_.tolist()

    # Iterate over animal ids
    for aid in range(len(traj)):
        # append cluster label to animal id groups
        clust = [*km.labels_][aid]
        data_groups[[*data_groups.keys()
                     ][aid]] = data_groups[[*data_groups.keys()
                                            ][aid]].assign(cluster=clust)

        # append centroid of cluster to animal id groups
        data_groups[[*data_groups.keys()][aid]] = data_groups[[
            *data_groups.keys()
        ][aid]].assign(ClustCenter=clustcens[clust])

    # convert animal-id groups to dataframe
    clustered_df = regrouping_data(data_groups)

    clst = list(clustered_df.loc[:, "ClustCenter"])
    vars = list(zip(*clst))

    # append DataFrame with new centroid variables
    newvars = {}
    for i in range(len(varlst)):
        newvars['centroid_' + str(varlst[i])] = vars[i]
    clustered_df = clustered_df.join(pd.DataFrame(newvars))

    # if true, return inertia along with dataframe, else (default) just dataframe.
    if inertia:
        return clustered_df, km.inertia_

    else:
        return clustered_df


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
        preprocessed_data = medoid_computation(preprocessed_data)
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


def compute_polarization(preprocessed_data):
    """
    Compute the polarization of a group at all record timepoints.

    More info about the formula: Here: https://bit.ly/2xZ8uSI and Here: https://bit.ly/3aWfbDv.
    :param preprocessed_data: Pandas Dataframe with or without previously extracted features.
    :return: Pandas Dataframe, with extracted features along with a new "polarization" variable.
    """
    # Extract features if not done yet
    if "direction" not in preprocessed_data.columns:
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
    return polarization_data


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


def voronoi_diagram(preprocessed_data):
    """
    Compute the voronoi diagram for each time step as well as the area for each cell over time

    Each timestep gets a voronoi object as well as the area of the voronoi - shape.
    Infinity, if respective animal is outmost in swarm.

    Note: Voronoi object contains the following attributes:
        `.points` - Coordinates of input points.
        `.vertices` - Coordinates of the Voronoi vertices.
        `ridge_points` - Indices of the points between which each Voronoi ridge lies.
        `ridge_vertices` - Indices of the Voronoi vertices forming each Voronoi ridge.
        `regions` - Indices of the Voronoi vertices forming each Voronoi region. -1 indicates vertex outside the Voronoi
        diagram.
        `point_region` - Index of the Voronoi region for each input point. If qhull option “Qc” was not specified,
        the list will contain -1 for points that are not associated with a Voronoi region.
        `furthest_site` - True if this was a furthest site triangulation and False if not.

    :param preprocessed_data: Animal movement records
    :return: movement records with voronoi area, and list of voronoi-diagram objects for each timestep
    """

    data_time = preprocessed_data.groupby('time')

    # Dictionary to hold grouped data by 'time' attribute-
    data_groups_time = {}

    # List for diagram objects at each timepoint
    diagrams = []

    # Obtain diagram objects, store in list for each timestep
    for aid in data_time.groups.keys():
        data_groups_time[aid] = data_time.get_group(aid)
        data_groups_time[aid].reset_index(drop=True, inplace=True)
        diagrams.append(Voronoi(data_groups_time[aid].loc[:, ["x", "y"]]))

        # Calculate area based on voronoi-volumes function right above
        vor_vol = voronoi_volumes(data_groups_time[aid].loc[:, ["x", "y"]])

        # Store area values in new variable for each timestep
        data_groups_time[aid] = data_groups_time[aid].assign(
            area_voronoi=vor_vol)

    # Regroup data into DataFrame
    out_data = regrouping_data(data_groups_time)

    return out_data, diagrams
