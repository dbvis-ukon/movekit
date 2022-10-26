MOVEKIT
======

<p align="center">
    <img src="media/mover.gif" height=150px/>
    <img src="media/voronoi.png" height=150px>
    <img src="media/network.png" height=150px>
    <img src="media/movement.png" height=150px>
    <img src="media/movebank.png" height=150px>
    <img src="media/clustering.png" height=150px>
</p>


`movekit` is an open-source software package for the processing and analysis of movement data.

__`movekit` supports different tasks:__

* Data pre-processing
  * Check data for variables required for movement analysis
  * Clean data (remove duplicates, drop missing values, etc.)
  * Use interpolation to deal with missing values in data
  * Normalize the data
  * Filter the data
  * Many more preprocessing methods
* Feature extraction:
  * Extract different features such as the distance covered, the average speed, the average acceleration, the direction or the change of direction for individual movers in between every time step
  * Apply time series analysis on these extracted features (f.e. mean and variance of speed for mover over time)
  * Check different distances (euclidean & hausdorff) in between movers for different time steps
  * Detect outliers in data
  * Examine time and duration of different phases of movement (moving and stopping) for each mover
  * Analyze environment exploration of each mover
  * Many more features can be extracted
* Group-level analysis
  * Calculate centroids and medoids of the group of movers for different time steps
  * Calculate the centroid's movement direction and compare it with individual mover's directions
  * Compute polarization of movers
  * Identify different clusters in the data by applying various clustering algorithms
  * Obtain dynamic time warping of trajectories of the different movers
  * Many more group-level features can be extracted
* Spatial data analysis:
  * Create convex hull, voronoi diagram and delaunay triangulation for all movers at each time step
  * Extract areas of the created objects
* Network analysis with networkX
  * Create network graphs created for each time step and examine their attributes (centroid, polarization, total distance, mean speed, ...)
  * Investigate individual nodes of each time steps network graph
  * Investigate individual edges of each time steps network graph
  * Track development of network graphs over time
* Plotting analysis results:
  * Create basic plots for features such as acceleration or speed
  * Plot movement of movers in static or animated images
  * Create interactive map to plot geo data

`movekit` provides support for movement data and trajectories in different format:

__Data:__

* 2-dimensional data in the Euclidean space
* 3-dimensional data in the Euclidean space
* GPS coordinates (latitude and longitude)
* Data with different time formats
* Data in (Geo)JSON format
* Data from Movebank data base

---

## Installation

The easiest way to install *movekit* is by using `pip` :

    pip install movekit

---

## Demo

You can view a demo of common features here:
[Jupyter Notebooks](examples/).

---

### License

Released under a GNU General Public License. See the [LICENSE](LICENSE) file for details. List of [Authors](AUTHORS.rst)

The package is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy – EXC 2117 – 422037984.
