"""
  Microsoft Excel I/O in Python.
  Load the csv data into pandas dataframe.
  Author: Arjun Majumdar, Eren Cakmak
  Created: August, 2019
"""


import pandas as pd


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

            # Check if 'time' attribute is integer-
            if is_numeric_dtype(data['time']):
                data.sort_values('time', ascending=True, inplace=True)
            # Check if 'time' attribute is string-
            elif is_string_dtype(data['time']):
                data['time'] = pd.to_datetime(data['time'])
                data.sort_values('time', ascending=True, inplace=True)

        return data

    except FileNotFoundError:
        print(
            "Your file below could not be found. Please check path and/or file name and try again.\nPath given: {0}\n\n".format(path_to_file))

