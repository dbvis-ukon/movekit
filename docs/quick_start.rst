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

.. code-block:: python

   import movekit as mkit
   
   path = 'path/to/your/data'
   data = mkit.read_data(path)

*****
Preprocessing general method
*****

Preprocess your data with options to drop columns with missing values or interpolate them with various methods. For interpolation one can define certain parameters like the maximum number of missing values to fill or the method to use.

.. code-block:: python

   clean_data = mkit.preprocess(data, dropna=True, interpolation=False, limit=1, limit_direction='forward', inplace=False, method='linear', order=1, date_format=False)

*****
Some additional methods to reduce data size
*****

Additionally there exist some methods to reduce the size of the data. For example one can filter the data and only analyze a specific time period.

.. code-block:: python

    filtered_data = mkit.filter_dataframe(data, frm, to)

Another option is to apply sampling to the data. This can be done either systematically or randomly.

.. code-block:: python

    sampled_data = mkit.resample_systematic(data_groups, downsample_size)
    sampled_data = mkit.resample_random(data_groups, downsample_size)

It might be useful to split the entire data frame into different smaller sub data frames for each animal.

.. code-block:: python

    dict_with_diff_dataframes = mkit.split_trajectories(data_groups, segment, fuzzy_segment=0, csv=False)

*****
Methods to replace/convert specific values (duplicates, missings, selected values)
*****

One can easily replace/convert specific values in the data (missings, duplicates, selected values).
For example one can replace the coordinate values for a specific mover at a specific time period. This can be useful method to deal with outliers.

.. code-block:: python

    replaced_data_groups = mkit.replace_parts_animal_movement(data_animal_id_groups, animal_id, time_array,replacement_value_x, replacement_value_y, replacement_value_z=None)

In many applications it is useful to normalize the data for the coordinates before the analysis.

.. code-block:: python

    normalized_data = mkit.normalize(data)

One can not only normalize, but also scale the coordinates data such that it is between a specified min and max value.

.. code-block:: python

    scaled_data = mkit.convert_measueres(preprocessed_data, x_min = 0, x_max = 1, y_min = 0, y_max = 1, z_min = 0, z_max = 1)

With missing data can be dealt using interpolation (see also general method `preprocess` above).

.. code-block:: python

    interpolated_data = mkit.interpolate(data,limit=1,limit_direction="forward",inplace=False,method="linear",order=1, date_format=False)

To get an overview over the missing data there are two methods one can apply.

.. code-block:: python

    mkit.print_missing(data)
    mkit.plot_missing_values

Also rows which contain duplicates can be explored.

.. code-block:: python

    mkit.print_duplicate(data)

If specific movers are not of interest for the analysis, they can be removed.

.. code-block:: python

    mkit.delete_mover(data, animal_id)

*****
Making a pandas DataFrame compatible with movekit
*****
If one has the data stored in a Pandas DataFrame one can easily make the DataFrame compatible with movekit by giving the `from_dataframe` function a dictionary to map the column names from the existing DataFrame to be compatible with the required column names by movekit.

.. code-block:: python

    mkit.from_dataframe(data, dictionary)

*****
Support for geographic coordinates
*****
Additionally movekit is able to project data from GPS coordinates in the latitude and longitude format to the cartesian coordinate system. By giving the function as input the names of the columns storing the geographic coordinates it converts the coordinates to a cartesian coordinate system.

.. code-block:: python

    mkit.convert_latlon(data, latitude='latitude', longitude='longitude', replace=True)



