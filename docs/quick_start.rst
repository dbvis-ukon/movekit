Quick Start
===========

Installation
------------

To use movekit, first install it using pip:

.. code-block:: console

   $ pip install movekit

First Steps
-----------

*****
Read in data
*****

Read in data. Supported formats include `csv` and `xlsx` and `xls`. The function will return a pandas DataFrame, the native data structure to movekit.

**read_data(path, sheet=0, time_format="undefined")**:
    | Function to import data from 'csv', 'xlsx' and 'xls' files.
    | param path: Complete path/relative path to Excel file along with file name.
    | param sheet: name of specific sheet given, by default first sheet of the excel workbook.
    | param time_format: As the time is converted to a datetime object, the time format has to be indicated for unusual time formats. For the different time formats refer to: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
    | return: Pandas DataFrame containing imported data.

.. code-block:: python

   import movekit as mkit
   
   path = 'path/to/your/data'
   data = mkit.read_data(path)
   # data is not an integer but in time format DD-MM-YYYY:
   data = mkit.read_data(path, time_format = '%d-%m-%Y')

Additionally movekit is able to read data files containing data and geometry, such as GeoPackage, (Geo)json, Shapefile and many more.
When importing the data one can define whether the movers' coordinates should be converted from the geometry object or if they are already
given in the data additionally to the geometry.

**read_with_geometry(path, animal_id="name", time="time", x="x", y="y",  z="z", coordinate_conversion=False, time_format="undefined", geopandas = True)**:
    | Function to import files containing both data and geometry (f.e. GeoPackage, (Geo)json, Shapefile and many more).
    | param path: Complete path/relative path to file along with file name
    | param animal_id: Key name of the unique animal identifier (as f.e. defined as property value in the geojson feature)
    | param time: Key name of time (as defined f.e. as property value in the geojson feature)
    | param x: Key name of x variable (as defined f.e. as property value in the geojson feature)
    | param y: Key name of y variable (as defined f.e. as property value in the geojson feature)
    | param z: Key name of z variable (as defined f.e. as property value in the geojson feature)
    | param coordinate_conversion: Boolean defining whether the x,y (and z) coordinates are stored in geometry object of imported data (f.e. as geometry value in geojson feature). Note that if coordinate_conversion=True function searches for point object in geometry column and converts coordinates to individual columns of returned data frame. (In case of multiple geometries for an observation, so called 'geometry collections', the first point object is converted.)
    | param time_format: If time is given in an unusual format, the format has to be indicated for the conversion.
    | param geopandas: Boolean defining whether the returned data frame is a geopandas data frame (containing geometry objects in column 'geometry') or a pandas data frame (not containing a 'geometry' column).
    | return: Geopandas or pandas data frame in a format required for using the movekit package.

.. code-block:: python

    data = mkit.read_with_geometry('file_path')

Movekit can also import animal movement data from the Movebank database. Note that there is also a notebook in the documentation showing some exemplary analysis for Movebank data.

**read_movebank(path_to_file, animal_id = 'individual-local-identifier')**:
    | Function to import csv and excel files from the Movebank database.
    | param path_to_file: Complete path/relative path to file along with file name.
    | param animal_id: Column name of the unique animal identifier (converted to be animal_id).
    | return: Data frame in a format required for using the movekit package.

.. code-block:: python

    data = mkit.read_movebank('file_path')

*****
Preprocessing general method
*****

Preprocess your data with options to drop columns with missing values or interpolate them with various methods. For interpolation one can define certain parameters like the maximum number of missing values to fill or the method to use.

**preprocess(data,dropna=True,interpolation=False,limit=1,limit_direction="forward",inplace=False,method="linear",order=1,date_format = False)**:
    | Function to perform data preprocessing. Print the number of missing values per column; Drop columns with missing values for 'time' and 'animal_id'; Remove the duplicated rows found.
    | param data: DataFrame to perform preprocessing on
    | param dropna: Optional parameter to drop columns with  missing values for 'time' and 'animal_id'
    | param interpolation: Optional parameter to perform interpolation
    | param limit: Maximum number of consecutive NANs to fill
    | param limit_direction: If limit is specified, consecutive NaNs will be filled in this direction.
    | param inplace: Update the  data in place if possible.
    | param method: Interpolation technique to use. Default is "linear".
    | param order: To be used in case of polynomial or spline interpolation.
    | param date_format: Boolean to define whether time is some kind of date format. Important for interpolation.
    | return: Preprocessed DataFrame.

.. code-block:: python

   clean_data = mkit.preprocess(data, dropna=True, interpolation=False, limit=1, limit_direction='forward', inplace=False, method='linear', order=1, date_format=False)

*****
Some additional methods to reduce data size
*****

Additionally there exist some methods to reduce the size of the data. For example one can filter the data and only analyze a specific time period.

**filter_dataframe(data, frm, to)**:
    | Extract records of assigned time frame from preprocessed movement record data.
    | param data: Pandas DataFrame, containing preprocessed movement record data.
    | param frm: Int, defining starting point from where to extract records.Note that if time is stored as a date (if input data has time not stored as numeric type it is automatically converted to datetime) parameter has to be set using an datetime format: mkit.filter_dataframe(data, "2008-01-01", "2010-10-01")
    | param to: Int, defining end point up to where to extract records.
    | return: Pandas DataFrame, filtered by records matching the defined frame in 'from'-'to'.

.. code-block:: python

    filtered_data = mkit.filter_dataframe(data, frm, to)

Another option is to apply sampling to the data. This can be done either systematically or randomly.

**resample_systematic(data_groups, downsample_size)**:
    | Resample the movement data of each animal - by downsampling at fixed time intervals.
    | This is done to reduce the resolution of the dataset. This function does this by systematically choosing samples from each animal.
    | param data_groups: DataFrame containing the movement records.
    | param downsample_size: Int sample size to which each animal has to be reduced by downsampling.
    | return: DataFrame, modified from original size 'data_groups' to 'downsample_size'.

**resample_random(data_groups, downsample_size)**:
    | Resample the movement data of each animal - by downsampling at random time intervals.
    | This is done to reduce resolution of the dataset. This function does this by randomly choosing samples from each animal.
    | param data_groups: DataFrame containing the movement records.
    | param downsample_size: Int sample size to which each animal has to be reduced by downsampling.
    | return: DataFrame, modified from original size 'data_groups' to 'downsample_size'.

.. code-block:: python

    sampled_data = mkit.resample_systematic(data_groups, downsample_size)
    sampled_data = mkit.resample_random(data_groups, downsample_size)

It might be useful to split the entire data frame into different smaller sub data frames for each animal.

**split_trajectories(data_groups, segment, fuzzy_segment=0, csv=False)**:
    | Split trajectory of a single animal into several segments based on specific criterion.
    | param data_groups: DataFrame with movement records.
    | param segment: Int, defining point where the animals are split into several Pandas Data Frames.
    | param fuzzy_segment: Int, defining interval which will overlap on either side of the segments.
    | param csv: Boolean, defining if each interval shall be exported locally as singular csv
    | return: Dictionary with the created DataFrames for each animal.

.. code-block:: python

    dict_with_diff_dataframes = mkit.split_trajectories(data_groups, segment, fuzzy_segment=0, csv=False)

*****
Methods to replace/convert specific values (duplicates, missings, selected values)
*****

One can easily replace/convert specific values in the data (missings, duplicates, selected values).
For example one can replace the coordinate values for a specific mover at a specific time period. This can be useful method to deal with outliers.

**replace_parts_animal_movement(data_groups, animal_id, time_array, replacement_value_x, replacement_value_y, replacement_value_z=None)**:
    | Replace subsets (segments) of animal movement based on some indices e.g. time.
    | This function can be used to remove outliers.
    | param data_groups: DataFrame containing the movement records.
    | param animal_id: Int defining 'animal_id' whose movements have to be replaced.
    | param time_array: Array defining time indices whose movements have to replaced (array of integers if time has integer format, array of strings with datetime if time is datetime format)
    | param replacement_value_x: Int value that will replace all 'x' attribute values in 'time_array'.
    | param replacement_value_y: Int value that will replace all 'y' attribute values in 'time_array'.
    | param replacement_value_z: Int value that will replace all 'z' attribute values in 'time_array'. (optional)
    | return: Dictionary with replaced subsets.

.. code-block:: python

    replaced_data_groups = mkit.replace_parts_animal_movement(data_animal_id_groups, animal_id, time_array,replacement_value_x, replacement_value_y, replacement_value_z=None)

In many applications it is useful to normalize the data for the coordinates before the analysis.

**normalize(data)**:
    | Normalizes values for the 'x' and 'y' column
    | param data: DataFrame to perform preprocessing on
    | return: normalized DataFrame

.. code-block:: python

    normalized_data = mkit.normalize(data)

One can not only normalize, but also scale the coordinates data such that it is between a specified min and max value.

**convert_measueres(preprocessed_data, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1)**:
    | Create a linear scale with input parameters for x,y for transformation of position data.
    | param preprocessed_data: Pandas DataFrame only with x and y position data
    | param x_min: int minimum for x - default: 0.
    | param x_max: int maximum for x - default: 1.
    | param y_min: int minimum for y - default: 0.
    | param y_max: int maximum for y - default: 1.
    | param z_min: int minimum for z - default: 0.
    | param z_max: int maximum for z - default: 1.
    | return: Pandas DataFrame with linearly transformed position data.

.. code-block:: python

    scaled_data = mkit.convert_measueres(preprocessed_data, x_min = 0, x_max = 1, y_min = 0, y_max = 1, z_min = 0, z_max = 1)

With missing data can be dealt using interpolation (see also general method `preprocess` above).

**interpolate(data, limit=1,limit_direction="forward",inplace=False,method="linear",order=1,date_format=False)**:
    | Interpolate over missing values in pandas Dataframe of movement records.
    | Interpolation methods consist of "linear", "polynomial, "time", "index", "pad". (see  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html)
    | param data: Pandas DataFrame of movement records
    | param limit: Maximum number of consecutive NANs to fill
    | param limit_direction: If limit is specified, consecutive NaNs will be filled in this direction.
    | param inplace: Update the data in place if possible.
    | param method: Interpolation technique to use. Default is "linear".
    | param order: To be used in case of polynomial or spline interpolation.
    | param date_format: Boolean to define whether time is some kind of date format. In this case column type has to be converted before calling interpolate.
    | return: Interpolated DataFrame.

.. code-block:: python

    interpolated_data = mkit.interpolate(data,limit=1,limit_direction="forward",inplace=False,method="linear",order=1, date_format=False)

To get an overview over the missing data there are two methods one can apply.

**print_missing(df)**:
    | Print the missing values for each column.
    | param df: Pandas DataFrame of movement records.
    | return: None.

**plot_missing_values(data)**:
    | Plot the missing values of an animal-ID against time.
    | param data: Pandas DataFrame containing records of movement.
    | return: None.

.. code-block:: python

    mkit.print_missing(data)
    mkit.plot_missing_values

Also rows which contain duplicates can be explored.

**print_duplicate(df)**:
    | Print rows, which are duplicates.
    | param df: Pandas DataFrame of movement records.
    | return: None.

.. code-block:: python

    mkit.print_duplicate(data)

If specific movers are not of interest for the analysis, they can be removed.

**delete_mover(data, animal_id)**:
    | Delete a particular mover from the DataFrame
    | param data: DataFrame
    | param animal_id: int. The animal_id as found in the column animal_id
    | return: DataFrame

.. code-block:: python

    mkit.delete_mover(data, animal_id)

*****
Making a pandas DataFrame compatible with movekit
*****
If one has the data stored in a Pandas DataFrame one can easily make the DataFrame compatible with movekit by giving the `from_dataframe` function a dictionary to map the column names from the existing DataFrame to be compatible with the required column names by movekit.

**from_dataframe(data, dictionary)**:
    | Reformat an existing DataFrame to make it compatible with movekit
    | param data: pandas DataFrame. The data to be reformatted
    | param dictionary: Key-value pairs of column names. Keys store the old column names. The respective new column names are stored as their values. Values that need to be defined include 'time', 'animal_id', 'x' and 'y'
    | return: pandas DataFrame

.. code-block:: python

    mkit.from_dataframe(data, dictionary)

*****
Support for geographic coordinates
*****
Additionally movekit is able to project data from GPS coordinates in the latitude and longitude format to the cartesian coordinate system. By giving the function as input the names of the columns storing the geographic coordinates it converts the coordinates to a cartesian coordinate system.

**convert_latlon(data, latitude='latitude', longitude='longitude', replace=True)**:
    | Project data from GPS coordinates (latitude and longitude) to the cartesian coordinate system
    | param data: DataFrame with GPS coordinates
    | param latitude: str. Name of the column where latitude is stored
    | param longitude: str. Name of the column where longitude is stored
    | param replace: bool. Flag whether the xy columns should replace the latlon columns
    | return: DataFrame after the transformation where latitude is projected into y and longitude is projected into x

.. code-block:: python

    mkit.convert_latlon(data, latitude='latitude', longitude='longitude', replace=True)

