Extracting features
===========

One can easily calculate features such as distance, direction, average speed, average acceleration, turning and stops of the mover with the `extract_features` function.

**extract_features(data, fps=10, stop_threshold=0.5)**:
    |Calculate and return all absolute features for input animal group.
    |param data: pandas DataFrame with all records of movements.
    |param fps: size of window used to calculate average speed and average acceleration:
    integer to define size of window for integer-formatted time or string to define size of window for datetime-formatted time (For possible units refer to: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.)
    |param stop_threshold: integer to specify threshold for average speed, such that we consider time step a "stop".
    |return: pandas DataFrame with additional variables consisting of all relevant features.
    |Notes to extracted features:
    |*Distance*: returns metric distance in between two time steps.
    |*Direction*: returns movement vector for each time step by checking the difference of the coordinates to the previous time step.
    |*Average speed*: Compute average speed of mover based on fps parameter. The formula used for calculating average speed is: (Total Distance traveled) / (Total time taken).
    |Size of traveling window is determined by fps parameter:
    |By choosing f.e. fps=4 at time step 5: (distance covered from time step 3 to time step 7) / 4.
    |By choosing f.e. fps=3 at time step 5: (distance covered from time step 3.5 to time step 6.5) / 3. (in this case use of interpolation if time steps 3.5 and 6.5 do not exist.)
    |*Average acceleration*: Compute average acceleration of mover based on fps parameter. The formula used for calculating average acceleration is: (Final Speed - Initial Speed) / (Total Time Taken).
    |Size of traveling window is determined by fps parameter:
    |By choosing f.e. fps=4 at time step 5: (speed at time step 7 - speed at time step 3) / 4.
    |By choosing f.e. fps=3 at time step 5: (speed at time step 6.5 - speed at time step 3.5) / 3. (in this case use of interpolation if time steps 3.5 and 6.5 do not exist.)
    |*Turning*: Computes the turning for a mover between two time steps as the cosine similarity between its direction vectors.
    |*Stops*: Defines a record as 'Stopped' where the value is 1 if 'Average_Speed' <= threshold_speed and 0 otherwise.

.. code-block:: python

   data_features = mkit.extract_features(data, fps = 10, stop_threshold = 0.5)
   # example for datetime-formatted time with window size of 5 seconds
   data_features = mkit.extract_features(data, fps = '5S', stop_threshold = 0.5)

For 2-dimensional data one can also extract the direction and turning angle for a mover between two timesteps by using `compute_direction_angle` and `compute_turning_angle`.

**compute_direction_angle(data, param_x='x', param_y='y', colname='direction_angle')**:
    |Computes the angle of rotation of an animal between two timesteps. Only possible if coordinates are 2D only.
    |param data: dataframe containing the movement records
    |param param_x: column name of the x coordinate
    |param param_y: column name of the y coordinate
    |param colname: the name to appear in the new DataFrame for the direction angle computed.
    |return: dataframe containing computed 'direction_angle' as angle from 0-360 degrees (x-axis to the right is 0 degrees)

**compute_turning_angle(data, colname='turning_angle', direction_angle_name='direction_angle')**:
    |Computes the turning angle for a mover between two timesteps as the difference of its direction angle. Only possible for 2D data.
    |param data: dataframe containing the movement records.
    |param colname: the name of the new column to be added.
    |param direction_angle_name: the name of the column containg the direction angle for each movement record.
    |return: dataframe containing an additional column with the difference in degrees between current and previous time step for each record.
    |Note that difference can not be higher than +-180 degrees.

.. code-block:: python

   data_direction_angle = mkit.direction_angle(data)
   data_turning_angle = mkit.turning_angle(data)

*****
Computing distances between movers and between time steps
*****
Distances between different movers can easily be computed. For example the euclidean distance is computed with the function `euclidean_dist`.

**euclidean_dist(data)**:
    |Compute the euclidean distance between movers for one individual grouped time step using the Scipy 'pdist' and 'squareform' methods.
    |param data: Preprocessed pandas DataFrame with positional record data containing no duplicates.
    |return: pandas DataFrame, including computed euclidean distances.

.. code-block:: python

   distances = mkit.euclidean_dist(data)

Also one can analyze the distance between the different positions of each mover for a particular time window.

**distance_by_time(data, frm, to)**:
    |Computes the distance between positions for a particular time window for all movers.
    |param data: pandas DataFrame with all records of movements.
    |param frm: int defining the start of the time window. Note that if time is stored as a date (if input data has time not stored as numeric type it is automatically converted to datetime) parameter has to be set using an datetime format: mkit.distance_by_time(data, "2008-01-01", "2010-10-01")
    |param to: Int, defining end point up to where to extract records.
    |param to: int defining the end of the time window (inclusive)
    |return: pandas DataFrame with animal_id and distance

.. code-block:: python

   distances = mkit.distance_by_time(data, frm, to)

Additionally one can obtain a matrix of the trajectory similarities, based on the Hausdorff distance of trajectories of the animals with the function `hausdorff_distance`.

**hausdorff_distance(data, mover1=None, mover2=None)**:
    |Calculate the Hausdorff-Distance between trajectories of different movers.
    |param data: pandas DataFrame containing movement records.
    |param mover1: animal_id of the first mover if Hausdorff distance is just to be calculated between two movers.
    |param mover2: animal_id of the second mover if Hausdorff distance is just to be calculated between two movers
    |return: Hausdorff distance between two specified movers. If no movers are specified, Hausdorff distance between all movers in the data to each other as a Pandas DataFrame.

.. code-block:: python

   mkit.hausdorff_distance(data)

*****
Computing centroids and medoids for each time stamp
*****
With `centroid_medoid_computation` the centroids, the medoids and the distances of each mover to the centriod can be calculated for each time stamp.

**centroid_medoid_computation(data, only_centroid=False, object_output=False)**:
    |Calculates the data point (animal_id) closest to center/centroid/medoid for a time step
    |param data: Pandas DataFrame containing movement records
    |param only_centroid: Boolean in case we just want to compute the centroids. Default: False.
    |param object_output: Boolean whether to create a point object for the calculated centroids. Default: False.
    |return: Pandas DataFrame containing computed medoids & centroids

.. code-block:: python

   centroid_medoid_computation(data, only_centroid=False, object_output=False)

*****
Exploring the geospatial features and plotting the data
*****
Furthermore plots can easily be created, such as the movement from all movers in a specified time period or the movements from individual movers.

**plot_movement(data, frm, to)**:
    |Plot 'x' and 'y' attributes for given Pandas DataFrame in specified time frame.
    |param data: Pandas DataFrame (should be sorted by 'time' attribute).
    |param frm: Starting from time step. Note that if time is stored as a date (if input data has time not stored as numeric type it is automatically converted to datetime) parameter has to be set using an datetime format: mkit.plot_movement(data, "2008-01-01", "2010-10-01")
    |param to: Ending to time step.
    |return: None.

**plot_animal(inp_data, animal_id)**:
    |Plot individual animal's 'x' and 'y' coordinates.
    |param inp_data: DataFrame containing 'x' & 'y' attributes.
    |param animal_id: ID of animal to be plotted.
    |return: None.

.. code-block:: python

    mkit.plot_movement(data, frm, to)
    mkit.plot_animal(inp_data, animal_id)

Also animations of the movements from the different movers can be displayed and saved as gif.

**animate_movement(data, viewsize)**:
    |Animated version of plot_movement function.
    |Animates 'x' and 'y' attributes for given Pandas DataFrame in specified time frame.
    |param data: Pandas DataFrame (should be sorted by 'time' attribute).
    |param viewsize: Int. Define how many time steps/frames should be visible in the animation.

**save_animation_plot(animation_object, filename)**:
    |save animation as gif file in working directory.
    |param animation_object: created animation object
    |param filename: name of the file which is created

.. code-block:: python

    anim = mkit.animate_movement(data, 100)
    mkit.save_animation_plot(anim, 'filename')

One can also plot either the average acceleration or the average speed for each individual mover/animal over time.

**plot_pace(avg_speed_data, feature="speed")**:
    |Plot average speed or average acceleration extracted feature for each animal.
    |param avg_speed_data: pandas Dataframe including average speed feature.
    |param feature: either 'speed' or 'acceleration'
    |return: None.

.. code-block:: python

    mkit.plot_pace(data_features, "speed")

One can additionally check the geospatial distribution of the different movers. The function `explore_features_geospatial` shows the exploration of environment space by each animal. It gives singular descriptions of polygon area covered by each animal and combined.

**explore_features_geospatial(preprocessed_data)**:
    |Show exploration of environment space by each animal using 'shapely' package.
    |Gives singular descriptions of polygon area covered by each animal and combined. Additionally a plot of the respective areas is provided.
    |param preprocessed_data: pandas DataFrame, containing preprocessed movement records.
    |return: None.

.. code-block:: python

    mkit.explore_features_geospatial(data)

To get the percentage environment space explored by singular animal one can use `explore_features`.

**explore_features(data)**:
    |Show percentage of environment space explored by singular animal.
    |Using minumum and maximum of 2-D coordinates, given by 'x' and 'y' features in input DataFrame.
    |param data: pandas DataFrame, containing preprocessed movement records.
    |return: None.

.. code-block:: python

    mkit.explore_features(data)

To examine the number of time steps for each mover id one can call `plot_animal_timesteps`.

**plot_animal_timesteps(data)**:
    |Plot the number of time steps for each 'animal_id'
    |param data_animal_id_groups: DataFrame containing movement records.
    |return: None

.. code-block:: python

    mkit.plot_animal_timesteps(data)

Geodata can be plotted on an interactive map by calling `plot_geodata`, afterwards it can be saved using `save_geodata_map`.

**plot_geodata(data, latitude_colname = "location-lat", longitude_colname = "location-long", animal_list=[], movement_lines=False)**:
    |Function to plot geo data on an interactive map using Open Street Maps.
    |param data: DataFrame containing the movement records
    |param latitude_colname: name of the column containing the latitude of each movement record
    |param longitude_colname: name of the column containing the longitude of each movement record
    |param animal_list: list containing animal_id's of all animals to be plotted (Default: every animal in data is plotted)
    |param movement_lines: Boolean whether movement lines between different location markers of animals are plotted
    |return: map Object containing markers for each tracked animal position

**save_geodata_map(map, filename)**:
    |save the created geodata map as a file
    |param map: map object to be saved.
    |param filename: name of the new created file containing the map.

.. code-block:: python

    map = mkit.plot_geodata(data)
    mkit.save_geodata(map, 'new_map')

*****
Splitting the trajectory of each animal in stopping and moving phases
*****
Movekit has a function to split the trajectories for each animal into moving and stopping phases according to a given stop threshold.
Additionally the durations of these individual phases can be examined. Both functions return a dictionary with animal ID as key.

**split_movement_trajectory(data, stop_threshold=0.5, csv=False)**:
    |Split trajectories of movers in stopping and moving phases.
    |param data: pandas DataFrame containing preprocessed movement records.
    |param stop_threshold: integer to specify threshold for average speed, such that we consider time step a "stop".
    |param csv: Boolean, defining if each phase shall be exported locally as singular csv.
    |return: dictionary with animal_id as key and list of individual dataFrames for each movement phase as values.

**movement_stopping_durations(data, stop_threshold=0.5)**:
    |Split trajectories of movers in stopping and moving phases and return the duration of each phase.
    |param data: pandas DataFrame containing preprocessed movement records.
    |param stop_threshold: integer to specify threshold for average speed, such that we consider time step a "stop".
    |return: dictionary with animal_id as key and DataFrame with the different phases and their durations as value.

.. code-block:: python

    mkit.split_movement_trajectory(data, stop_threshold = 0.5)
    mkit.movement_stopping_durations(data_features, stop_threshold = 0.5)

*****
Time series analysis
*****
Movekit also allows to extract many time series features by defining the required feature as parameter of the `ts_feature`. For a full list of all the features that can be extracted refer to https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html.

**ts_feature(data, feature)**:
    |Perform time series analysis by extracting specified time series features from record data.
    |param data: pandas DataFrame, containing preprocessed movement records and features.
    |param feature: time series feature which is extracted from the movement records.
    |return: pandas DataFrame, containing defined extracted time series features for each id for each feature.

**ts_all_features(data)**:
    |Perform time series analysis on record data.
    |param data: pandas DataFrame, containing preprocessed movement records and features.
    |return: pandas DataFrame, containing extracted time series features for each id for each feature.

.. code-block:: python

    mkit.ts_feature(data, feature)
    mkit.ts_all_feature(data)

