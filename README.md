MOVEKIT
======

<p align="center">
    <img src="media/mover.gif" height=150px/>
    <img src="media/voronoi.png" height=150px>
    <img src="media/network.png" height=150px> 
</p>


`movekit` is an open-source software package for the processing and analysis of movement data.

__Features:__

* Data pre-processing (e.g. data checks, smoothing, duplicate removal, interpolation, outlier detection)
* Feature extraction (e.g. speed, acceleration, heading)
* Individual-level movement analysis (e.g. autocorrelation, speed distribution, environment exploration)
* Group-level analysis (e.g. cohesion, polarisation, coordination, leadership, clustering)
* Spatial data analysis (Voronoi, delaunay triangulation)
* Network analysis with networkX

`movekit` provides support for use cases with movement data and trajectories:

__Data:__

* 2-dimensional data in the Euclidean space
* 3-dimensional data in the Euclidean space
* GPS coordinates (latitude and longitude)

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
