import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pandas.errors import EmptyDataError
import warnings

def parse_csv(path_to_file):
    """
    Read CSV file into Pandas DataFrame.

    :param path_to_file: Complete path/relative path to CSV file along with file name.
    :return: Pandas DataFrame containing imported data.
    """
    try:

        if path_to_file[-3:] == 'csv':
            data = pd.read_csv(path_to_file)
        else:
            data = pd.read_csv(path_to_file + '.csv')

        # change column names all to lower case values
        data.columns = map(str.lower, data.columns)

        format_col = ['time', 'animal_id', 'x', 'y']

        # check if all required columns are there in the right format
        if 'time' in data and 'animal_id' in data and 'x' in data and 'y' in data:
            # Check if 'time' attribute is integer-
            if is_numeric_dtype(data['time']):
                data.sort_values(['time', 'animal_id'],
                                 ascending=True,
                                 inplace=True)
                # Check if 'time' attribute is string-
            elif is_string_dtype(data['time']):
                data['time'] = pd.to_datetime(data['time'])
                data.sort_values(['time', 'animal_id'],
                                 ascending=True,
                                 inplace=True)
                data.drop(data.filter(regex="unname"), axis=1, inplace=True)
            # check if 'x' and 'y' are numeric-
            if not is_numeric_dtype(data['x']):
                warnings.warn(
                    "'x' attribute has to be of numeric data type for most of the analyses to be performed"
                )
            if not is_numeric_dtype(data['y']):
                warnings.warn(
                    "'y' attribute has to be of numeric data type for most of the analyses to be performed"
                )

            # Check if 'heading_angle' attribute is given in CSV file-
            if 'heading_angle' in data and np.issubdtype(
                    data['heading_angle'].dtype, np.number):
                print(
                    "\n'heading_angle' attribute is found and will be processed\n"
                )

            return data
        else:
            raise ValueError('Movekit requires columns to be named {} but was given {} instead'.format(format_col, data.columns))

    except FileNotFoundError:
        print("The file could not be found.\nPath given: {0}\n\n".format(
            path_to_file))



def parse_excel(path_to_file, sheet):
    """
    Read Excel file into Pandas DataFrame

    :param path_to_file: Complete path/relative path to Excel file along with file name
    :param sheet: name of specific sheet given, by default first sheet of the excel workbook
    :return: Pandas DataFrame containing imported data.
    """
    try:

        if path_to_file[-3:] == 'xls' or path_to_file[-4:] == 'xlsx':
            data = pd.read_excel(path_to_file, sheet)
        else:
            data = pd.read_excel(path_to_file + '.xlsx', sheet)

        if data.empty:
            raise EmptyDataError

        # change column names all to lower case values
        data.columns = map(str.lower, data.columns)

        format_col = ['time', 'animal_id', 'x', 'y']

        # check if all required columns are there in the right format
        if 'time' in data and 'animal_id' in data and 'x' in data and 'y' in data:
            # Check if 'time' attribute is integer-
            if is_numeric_dtype(data['time']):
                data.sort_values(['time', 'animal_id'],
                                 ascending=True,
                                 inplace=True)
                # Check if 'time' attribute is string-
            elif is_string_dtype(data['time']):
                data['time'] = pd.to_datetime(data['time'])
                data.sort_values(['time', 'animal_id'],
                                 ascending=True,
                                 inplace=True)
            # check if 'x' and 'y' are numeric-
            if not is_numeric_dtype(data['x']):
                warnings.warn(
                    "'x' attribute has to be of numeric data type for most of the analyses to be performed"
                )
            if not is_numeric_dtype(data['y']):
                warnings.warn(
                    "'y' attribute has to be of numeric data type for most of the analyses to be performed"
                )

            # Check if 'heading_angle' attribute is given in CSV file-
            if 'heading_angle' in data and np.issubdtype(
                    data['heading_angle'].dtype, np.number):
                print(
                    "\n'heading_angle' attribute is found and will be processed\n"
                )
            return data
        else:
            raise ValueError('Movekit requires columns to be named {} but was given {} instead'.format(format_col, data.columns))

    except FileNotFoundError:
        print("The file could not be found.\nPath given: {0}\n\n".format(
            path_to_file))


def read_data(path, sheet = 0):
    """
    Function to import data from 'csv', 'xlsx' and 'xls' files.

    :param path: Complete path/relative path to Excel file along with file name
    :param sheet: name of specific sheet given, by default first sheet of the excel workbook
    :return: Pandas DataFrame containing imported data.
    """
    # Call appropriate IO function based on file extension
    # Split string based on '.' (dot)-
    file_split = path.split(".")

    if file_split[-1] == 'csv':

        return parse_csv(path)
    elif file_split[-1] == 'xlsx':
        return parse_excel(path, sheet)
    elif file_split[-1] == 'xls':
        return parse_excel(path, sheet)
    else:
        raise ValueError(f'File extension {file_split[-1]} can not be imported. Imported file has to be of type ".csv" or ".xlsx" or ".xls".')
