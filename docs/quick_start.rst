Quick Start
===========

Installation
------------

To use movekit, first install it using pip:

.. code-block:: console

   $ pip install movekit

First Steps
-----------

Read in data. Supported formats include `csv` and `xlsx`. The function will return a pandas DataFrame, the native data structure to movekit.

.. code-block:: python

   import movekit as mkit
   
   path = 'path/to/your/data'
   data = mkit.read_data(path)

Preprocess your data with options to drop missing values or interpolate them with various methods.

.. code-block:: python

   clean_data = mkit.preprocess(data, dropna=True, interpolation=False, limit=1, limit_direction='forward', inplace=False, method='linear')

One can easily calculate features such as distance, direction, speed, acceleration and stops with the `extract_features` function.

.. code-block:: python

   data_features = mkit.extract_features(clean_data, fps = 10)