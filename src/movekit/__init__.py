# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

from .io import read_data  # noqa: E402
from .preprocess import preprocess, filter_dataframe, replace_parts_animal_movement, resample_systematic, \
    resample_random,split_trajectories,convert_measueres, interpolate, print_missing, print_duplicate, plot_missing_values  # noqa:
# E402
from .feature_extraction import extract_features, euclidean_dist  # noqa: E402
from .feature_extraction import ts_feature, ts_all_features, explore_features_geospatial, centroid_medoid_computation, \
    outlier_detection, group_movement
from .clustering import dtw_matrix, ts_cluster, get_heading_difference, compute_polarization, get_spatial_objects, compute_centroid_direction
# noqa: E402
from .plot import plot_animal, plot_pace, plot_movement  # noqa: E402

from .network import network_time_graphlist
