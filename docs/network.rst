Network analysis
===========

With the function `network_time_graphlist` one can calculate a network for each timestep based on delaunay triangulation. With the obtained network in depth analyses can be performed by applying the functionality of the `networkx` package. For further description refer to the example notebook.

**network_time_graphlist(preprocessed_data, object_type='delaunay_object', fps=10, stop_threshold=0.5)**:
    | Calculates a network list for each time step based on delaunay triangulation (currently only one available).
    | param preprocessed_data: Pandas DataFrame containing movement records.
    | param object_type: delaunay_object - currently the only one available.
    | param fps: as the returned network graph contains features such as average speed and average acceleration, the fps parameter defines the size of the travel window used to calculate these features. (refer to extract_features for a more detailed description of the parameter)
    | param stop_threshold: as the returned network graph contains a feature defining whether a timestamp is a stop, this parameter defines the average speed threshold for a timestamp to be a stop. (refer to extract_features for a more detailed description of the parameter)
    | return: List of nx graphs based on delaunay triangulation, containing singular, group and relational attributes on nodes, graph and edges.

.. code-block:: python

    graphs = mkit.network_time_graphlist(data)

