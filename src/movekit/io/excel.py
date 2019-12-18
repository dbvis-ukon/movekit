

import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
from pandas.api.types import is_numeric_dtype, is_string_dtype


"""
  Microsoft (MS) Excel I/O in Python.
  Load the MS Excel data into pandas dataframe.
  Author: Arjun Majumdar, Eren Cakmak
  Created: August, 2019
"""


def parse_excel(path_to_file):
    '''
    Function to read Excel file into a Pandas DataFrame-
    Expects complete path/relative path to CSV file along with file name

    Expects package 'xlrd' to be installed for this to work!
    '''
    try:

        if path_to_file[-3:] == 'xls' or path_to_file[-4:] == 'xlsx':
            data = pd.read_excel(path_to_file)
        else:
            data = pd.read_excel(path_to_file + '.xlsx')

        # change column names all to lower case values
        data.columns = map(str.lower, data.columns)

        # check if all required columns are there in the right format
        if 'time' in data and 'animal_id' in data and 'x' in data and 'y' in data:
            # Check if 'time' attribute is integer-
            if is_numeric_dtype(data['time']):
                data.sort_values('time', ascending=True, inplace=True)
                # Check if 'time' attribute is string-
            elif is_string_dtype(data['time']):
                data['time'] = pd.to_datetime(data['time'])
                data.sort_values('time', ascending=True, inplace=True)

            # Check if 'heading_angle' attribute is given in CSV file-
            if 'heading_angle' in data and np.issubdtype(data['heading_angle'].dtype, np.number):
                print("\n'heading_angle' attribute is found (numeric type) and will be processed\n")
                # do nothing, as 'heading_angle' attribute exists
            else:
                print("\nWARNING: 'heading_angle' attribute is not found in the given CSV data file. Continuing without it!\n")


            return data

    except FileNotFoundError:
        print(
            "Your file below could not be found.\nPath given: {0}\n\n".format(
                path_to_file))
    except EmptyDataError:
        print(
            'Your file is empty, has no header, or misses some required columns.'
        )