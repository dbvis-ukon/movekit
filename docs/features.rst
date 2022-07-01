Extracting features
===========

One can easily calculate features such as distance, direction, average speed, average acceleration, turning and stops of the mover with the `extract_features` function.

.. code-block:: python

   data_features = mkit.extract_features(data, fps = 10, stop_threshold = 0.5)

*****
Computing distances between movers and between timestamps
*****
Distances between different movers can easily be computed. For example the euclidean distance is computed with the function `euclidean_dist`.

.. code-block:: python

   distances = mkit.euclidean_dist(data)

Also one can analyze the distance between the different positions of each mover for a particular time window.

.. code-block:: python

   distances = mkit.distance_by_time(data, frm, to)

Additionally one can obtain a matrix of the trajectory similarities, based on the Hausdorff distance of trajectories of the animals with the function `hausdorff_distance`.

.. code-block:: python

   mkit.hausdorff_distance(data)

*****
Computing centroids and medoids for each time stamp
*****
With `centroid_medoid_computation` the centroids, the medoids and the distances of each mover to the centriod can be calculated for each time stamp.

.. code-block:: python

   centroid_medoid_computation(data, only_centroid=False, object_output=False)

*****
Exploring the geospatial features and plotting the data
*****
Furthermore plots can easily be created, such as the movement from all movers in a specified time period or the movements from individual movers.

.. code-block:: python

    mkit.plot_movement(data, frm, to)
    mkit.plot_animal(inp_data, animal_id)

Also animations of the movements from the different movers can be displayed and saved as gif and mp4.

.. code-block:: python

    anim = mkit.animate_movement(data, 100)
    mkit.save_animation_plot(anim, 'filename')

One can also plot either the average acceleration or the average speed for each individual mover/animal over time.

.. code-block:: python

    mkit.plot_pace(data_features, "speed")

One can additionally check the geospatial distribution of the different movers. The function `explore_features_geospatial` shows the exploration of environment space by each animal. It gives singular descriptions of polygon area covered by each animal and combined.

.. code-block:: python

    mkit.explore_features_geospatial(data)

*****
Splitting the trajectory of each animal in stopping and moving phases
*****
Movekit has a function to split the trajectories for each animal into moving and stopping phases according to a given stop threshold.
Additionally the durations of these individual phases can be examined. Both functions return a dictionary with animal ID as key.

.. code-block:: python

    mkit.split_movement_trajectory(data, stop_threshold = 0.5)
    mkit.movement_stopping_durations(data_features, stop_threshold = 0.5)

*****
Time series analysis
*****
Movekit also allows to extract many time series features by defining the required feature as parameter of the `ts_feature`. For a full list of all the features that can be extracted refer to https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html.

.. code-block:: python

    mkit.ts_feature(data, feature)
    mkit.ts_all_feature(data)

