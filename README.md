# MOVEKIT

**Simple and effective tools for the analysis of movement data**

*Movekit* is an open-source software package for the processing and analysis of movement data, including modules for:

 - Data pre-processing (e.g. data checks, smoothing, duplicate removal, interpolation, outlier detection)
 - Feature extraction (e.g. speed, acceleration, heading)
 - Individual-level movement analysis (e.g. autocorrelation, speed distribution, environment exploration)
 - Group-level analysis (e.g. cohesion, polarisation, coordination, leadership)

## Installation
The easiest way to install *movekit* is by using `pip` :

    pip install movekit

## Dependencies
- Python >=3.5
- Pandas (>=0.20.3, <=0.23.4)
- SciPy (>= 1.3.1)
- tsfresh (>= 0.12.0)
- xlrd (>= 1.2.0)
- seaborn (>= 0.9.0)

## Usage
You can view a demo of common features in this
[this Jupyter Notebook](/examples/demo.ipynb).

## Development
Movekit Development Status is 2-Pre-Alpha

For an overview of version changes see the [CHANGELOG](https://github.com/dbvis-ukon/movekit/blob/master/CHANGELOG).

Please submit bugs or feature requests to the GitHub issue tracker [here](https://github.com/dbvis-ukon/movekit/issues).

## License
This package was developed by Eren Cakmak, Arjun Majumdar, and Jolle Jolles from the [Data Analysis and Visualization Group](https://www.vis.uni-konstanz.de/) and the [Department of Collective Behaviour](http://collectivebehaviour.com) at the University Konstanz, with funding from the DFG Centre of Excellence 2117 "Centre for the Advanced Study of Collective Behaviour" (ID: 422037984).

Released under a GNU General Public License. See the [LICENSE](LICENSE) file for details.
