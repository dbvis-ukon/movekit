Spatial analysis
===========

With the function `get_spatial_objects` one obtains three types of spatial objects: Voronoi-Diagrams, Convex Hulls and Delaunay Triangles. Optionally, one may obtain only group-specific outputs - one object per time-capture. For a deeper discussion of the analyses that can be done using these objects refer to the example notebook.

.. code-block:: python

    spatial_obj = mkit.get_spatial_objects(data, group_output = True)

