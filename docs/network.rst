Network analysis
===========

With the function `network_time_graphlist` one can calculate a network for each timestep based on delaunay triangulation. With the obtained network in depth analyses can be performed by applying the functionality of the `networkx` package. For further description refer to the example notebook.

.. code-block:: python

    graphs = mkit.network_time_graphlist(data)

