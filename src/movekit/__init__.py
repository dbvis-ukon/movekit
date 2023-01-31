import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from .io import read_data, read_movebank, read_with_geometry  # noqa: E402
from .utils import presence_3d, angle
from .preprocess import preprocess, filter_dataframe, replace_parts_animal_movement, resample_systematic, \
    resample_random,split_trajectories,convert_measueres, interpolate, print_missing, print_duplicate, plot_missing_values, normalize, \
    delete_mover, convert_latlon, from_dataframe# noqa:
# E402
from .feature_extraction import extract_features, euclidean_dist, distance_by_time # noqa: E402
from .feature_extraction import ts_feature, ts_all_features, explore_features_geospatial, centroid_medoid_computation, \
    outlier_detection, group_movement, explore_features, compute_direction_angle, compute_turning_angle, split_movement_trajectory, movement_stopping_durations,hausdorff_distance, getis_ord
from .clustering import dtw_matrix, get_heading_difference, compute_polarization, get_spatial_objects, compute_centroid_direction, clustering, clustering_with_splits
# noqa: E402
from .plot import plot_animal, plot_pace, plot_movement, animate_movement, plot_geodata, save_geodata_map, save_animation_plot, plot_animal_timesteps, plot_heatmap # noqa: E402

from .network import network_time_graphlist
