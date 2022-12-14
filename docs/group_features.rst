Group Features
===========

*****
Detecting outliers
*****

One can perform detection of outliers, based on different outlier detection algorithms.

**outlier_detection(dataset, features=["distance", "average_speed", "average_acceleration", "stopped", "turning"], remove=False, algorithm="KNN", **kwargs)**:
    | Detect outliers based on different pyod algorithms. Note: User may decide on different parameters specific to algorithm chosen.
    | param dataset: Dataframe containing the movement records.
    | param features: list of features to detect outliers upon. Default: ["distance", "average_speed", "average_acceleration", "stopped", "turning"]
    | param remove: Boolean deciding whether outliers should be removed in returned dataframe. Default: False (outliers are not removed).
    | param algorithm: String defining which algorithm to use for finding the outliers. The following algorithms are available: "KNN", "ECOD", "COPOD", "ABOD", "MAD", "SOS", "KDE", "Sampling", "GMM", "PCA", "KPCA", "MCD", "CD", "OCSVM", "LMDD", "LOF", "COF", "CBLOF", "LOCI", "HBOS", "SOD", "ROD", "IForest", "INNE", "LSCP","LODA", "VAE", "SO_GAAL", "MO_GAAL", "DeepSVDD", "AnoGAN", "ALAD", "R-Graph".
    | Additional available algorithms: FastABOD: call algorithm="ABOD" with method="fast", AvgKNN: call algorithm="KNN" with method="mean", MedKNN: call algorithm="KNN" with method="median". Default algorithm is "KNN". For more information regarding all the algorithms refer to: https://pyod.readthedocs.io/en/latest/pyod.models.html#
    | param kwargs: Specific to the algorithm additional parameters can be specified. For further information about the algorithm-specific parameters once again refer to: https://pyod.readthedocs.io/en/latest/pyod.models.html#.
    | For the default algorithm "KNN" some additional parameters are for example:
    |   contamination - the contamination threshold, n_neighbors - the number of neighbors,
    |   method - which KNN to use, and metric - for distance computation.
    | return: Dataframe containing information for each movement record whether outlier or not.
.. code-block:: python

   mkit.outlier_detection(dataset, features=["distance", "average_speed", "average_acceleration", "stopped", "turning"], remove=False, algorithm="KNN", **kwargs)

Additionally the user can define threshold minimum and maximum values for different features to identify outliers.

**outlier_by_threshold(data, feature_thresholds, remove=False)**:
    | Identify outliers by user given features with specific minimum and maximum thresholds.
    | param data: data on which outliers are detected.
    | param feature_thresholds: dictionary containing the features as keys and the minimum/maximum threshold as two element list. For example if one would only want to declare all data points having an average speed < 5 and > 10 as outliers: feature_threshold = {'average_speed': [5,10]}
    | param remove: Boolean deciding whether outliers should be removed in returned dataframe. Default: False (outliers are not removed).
    | return: Dataframe containing information for each record whether it is an outlier according to the defined threshold values.

.. code-block:: python

    mkit.outlier_by_threshold(data, feature_thresholds, remove=False)

*****
Different analysis on group level
*****
For each time step one can obtain different group-level records. Records consist of total group-distance covered, mean speed, mean acceleration and mean distance from centroid for each time step. If input doesn't contain centroid or feature data, it is calculated, showing a warning.

**group_movement(feats)**:
    | Returns aggregated movement data, such as distance, mean speed, mean acceleration and mean distance to centroid for the entire group at each time capture.
    | param feats: pd DataFrame with animal-specific data - if no features contained, they will be extracted.
    | return: pd DataFrame with group-specific values for each time-capture

.. code-block:: python

    group_data = mkit.group_movement(data)

Once one has obtained centroids, medoids and distances to centroid from the different animals for each movement record with `centroid_medoid_computation`, one can continue with further analyses. For example one can calculate the centroid direction for each time step. Using this also the difference in the animal's direction and the centroid's direction for each timestep can be derived by computing the cosine similarity of the two direction vectors.

**compute_centroid_direction(data, colname="centroid_direction", group_output=False, only_centroid=True)**:
    | Calculate the direction of the centroid. Calculates centroid, if not in input data.
    | param pd DataFrame: DataFrame with x/y positional data and animal_ids, optionally include centroid
    | param colname: Name of the column. Default: centroid_direction.
    | param group_output: Boolean, defines form of output. Default: Animal-Level.
    | param only_centroid: Boolean in case we just want to compute the centroids. Default: True.
    | return: pandas DataFrame with centroid direction included

**get_heading_difference(preprocessed_data)**:
    | Calculate the difference in between the animal's direction and the centroid's direction for each timestep.
    | The difference is measured by the cosine similarity of the two direction vectors. The value range is from -1 to 1, with 1 meaning animal and centroid having the same direction while -1 meaning they have opposite directions.
    | param preprocessed_data: Pandas Dataframe containing preprocessed animal records.
    | return: Pandas Dataframe containing animal and centroid directions as well as the heading difference.

.. code-block:: python

    data = mkit.compute_centroid_direction(data,colname="centroid_direction",group_output=False,only_centroid=True)
    heading_diff = mkit.get_heading_difference(data)

Also the polarization of animals can be computed for each timestep. Value is between 0 and 1. More info about the formula used to calculate the polarization can be found on the following two links: https://bit.ly/2xZ8uSI and https://bit.ly/3aWfbDv. Note that if the data is three-dimensional, only the first two dimensions are considered to calculate the polarization.

**compute_polarization(preprocessed_data, group_output=False)**:
    | Compute the polarization of a group at all record timepoints.
    | More info about the formula: Here: https://bit.ly/2xZ8uSI and Here: https://bit.ly/3aWfbDv. As the formula only takes angles as input, the polarization is calculated for 2d - Data by first calculating the direction angles of the different movers and afterwards by calculating the polarization. For 3-dimensional data for all two's-combinations of the three dimensions the polarization is calculated in the way described before for 2d-data, afterwards the mean of the three results is taken as result for the polarization.
    | param preprocessed_data: Pandas Dataframe with or without previously extracted features.
    | return: Pandas Dataframe, with extracted features along with a new "polarization" variable.

.. code-block:: python

    pol = mkit.compute_polarization(data, group_output = False)

*****
Dynamic time warping
*****
Also a matrix to display the similarity of the animals trajectories based on the DTW algorithm can be computed.

**dtw_matrix(preprocessed_data, path=False, distance=euclidean)**:
    | Obtain dynamic time warping amongst all trajectories from the grouped animal-records.
    | param preprocessed_data: pandas Dataframe containing the movement records.
    | param path: Boolean to specify if matrix of dtw-path gets returned as well.
    | param distance: Specify which distance measure to use. Default: "euclidean". (ex. Alternatives: pdist, minkowski)
    | return: pandas Dataframe with distances between trajectories.

.. code-block:: python

    mkit.dtw_matrix(data)

