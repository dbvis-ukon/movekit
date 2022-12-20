Spatial analysis
===========

With the function `get_spatial_objects` one obtains three types of spatial objects: Voronoi-Diagrams, Convex Hulls and Delaunay Triangles. Optionally, one may obtain only group-specific outputs - one object per time-capture. For a deeper discussion of the analyses that can be done using these objects refer to the example notebook.

**get_spatial_objects(preprocessed_data, group_output=False)**:
    | Function to calculate convex hull, voronoi diagram and delaunay triangulation objects and also volumes of the first two objects.
    | Please visit https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/spatial.html for detailed documentation of spatial attributes.
    | param preprocessed_data: Pandas Df, containing x and y coordinates.
    | param group_output: Boolean, default: False, If true, one line per time capture for entire animal group.
    | return: DataFrame either for each animal or for group at each time, containing convex hull and voronoi diagram area as well as convex hull, voronoi diagram and delaunay triangulation object.

.. code-block:: python

    spatial_obj = mkit.get_spatial_objects(data, group_output = True)

