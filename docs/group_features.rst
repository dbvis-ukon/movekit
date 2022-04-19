Group Features
===========

*****
Detecting outliers
*****

One can perform detection of outliers, based on the KNN algorithm: user can define the regarding features for the detection, the number of the nearest neighbors taken into account for the outlier classification, the metric to calculate the distance, the method to aggregate the different distances, and the share of outliers.

.. code-block:: python

   mkit.outlier_detection(dataset, features=["distance", "average_speed", "average_acceleration", "stopped", "turning"], contamination=0.01, n_neighbors=5, method="mean", metric="minkowski")

*****
Different analysis on group level
*****
For each timestamp one can obtain different group-level records. Records consist of total group-distance covered, mean speed, mean acceleration and mean distance from centroid for each timestamp. If input doesn't contain centroid or feature data, it is calculated, showing a warning.

.. code-block:: python

    group_data = mkit.group_movement(data)

Once one has obtained centroids, medoids and distances to centroid from the different animals for each movement record with `centroid_medoid_computation`, one can continue with further analyses. For example one can calculate the centroid direction for each timestamp. Using this also the difference in the animal's direction and the centroid's direction for each timestep can be derived by computing the cosine similarity of the two direction vectors.

.. code-block:: python

    data = mkit.compute_centroid_direction(data,colname="centroid_direction",group_output=False,only_centroid=True)
    heading_diff = mkit.get_heading_difference(data)

Also the polarization of animals can be computed for each timestep. Value is between 0 and 1. More info about the formula used to calculate the polarization can be found on the following two links: https://bit.ly/2xZ8uSI and https://bit.ly/3aWfbDv. Note that if the data is three-dimensional, only the first two dimensions are considered to calculate the polarization.

.. code-block:: python

    pol = mkit.compute_polarization(data, group_output = False)

*****
Dynamic time warping
*****
Also a matrix to display the similarity of the animals trajectories based on the DTW algorithm can be computed.

.. code-block:: python

    mkit.dtw_matrix(data)

