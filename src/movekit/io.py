import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pandas.errors import EmptyDataError
import warnings
from .preprocess import convert_latlon
import geopandas as gpd
import shapely

def parse_csv(path_to_file, time_format, three_dim):
    """
    Read CSV file into Pandas DataFrame.
    :param path_to_file: Complete path/relative path to CSV file along with file name.
    :param time_format: If time is given in an unusual format, the format has to be indicated for the conversion.
    :param three_dim: Boolean defining whether data contains three dimensional coordinates
    :return: Pandas DataFrame containing imported data.
    """
    try:

        if path_to_file[-3:] == 'csv':
            data = pd.read_csv(path_to_file)
        else:
            data = pd.read_csv(path_to_file + '.csv')

        # change column names all to lower case values
        data.columns = map(str.lower, data.columns)

        if three_dim:
            format_col = ['time', 'animal_id', 'x', 'y', 'z']
        else:
            format_col = ['time', 'animal_id', 'x', 'y']

        # check if all required columns are there in the right format
        if three_dim:
            if 'time' in data and 'animal_id' in data and 'x' in data and 'y' in data and 'z' in data:
                contains_columns = True
            else:
                contains_columns = False
        else:
            if 'time' in data and 'animal_id' in data and 'x' in data and 'y' in data:
                contains_columns = True
            else:
                contains_columns = False
        if contains_columns:
            # Check if 'time' attribute is integer-
            if is_numeric_dtype(data['time']):
                data.sort_values(['time', 'animal_id'],
                                 ascending=True,
                                 inplace=True)
                # Check if 'time' attribute is string-
            elif is_string_dtype(data['time']):
                if time_format == "undefined":
                    data['time'] = pd.to_datetime(data['time'])
                    warnings.warn('As time was converted and no format was given as parameter, please check if the time conversion is correct.')
                else:
                    data['time'] = pd.to_datetime(data['time'], format=time_format)
                data.sort_values(['time', 'animal_id'],
                                 ascending=True,
                                 inplace=True)
                data.drop(data.filter(regex="unname"), axis=1, inplace=True)
            # check if 'x' and 'y' and 'z' are numeric-
            if not is_numeric_dtype(data['x']):
                warnings.warn(
                    "'x' attribute has to be of numeric data type for most of the analyses to be performed"
                )
            if not is_numeric_dtype(data['y']):
                warnings.warn(
                    "'y' attribute has to be of numeric data type for most of the analyses to be performed"
                )
            if three_dim:
                if not is_numeric_dtype(data['z']):
                    warnings.warn(
                        "'z' attribute has to be of numeric data type for most of the analyses to be performed"
                    )

            # Check if 'heading_angle' attribute is given in CSV file-
            if 'heading_angle' in data and np.issubdtype(
                    data['heading_angle'].dtype, np.number):
                print(
                    "\n'heading_angle' attribute is found and will be processed\n"
                )

            # Check if time contains duplicates and if so execute warning
            grouped_df = data.groupby("animal_id")
            for aid in grouped_df.groups.keys():
                if(grouped_df.get_group(aid)["time"].duplicated().any()):
                    warnings.warn(
                        "For some animals there are duplicate values for time! This might indicate some error in the "
                        "data and can lead to incorrect analysis."
                    )
                    break

            return data
        else:
            raise ValueError('Movekit requires columns to be named {} but was given {} instead'.format(format_col, data.columns))

    except FileNotFoundError:
        print("The file could not be found.\nPath given: {0}\n\n".format(
            path_to_file))



def parse_excel(path_to_file, sheet, time_format, three_dim):
    """
    Read Excel file into Pandas DataFrame
    :param path_to_file: Complete path/relative path to Excel file along with file name
    :param sheet: name of specific sheet given, by default first sheet of the excel workbook
    :param time_format: If time is given in an unusual format, the format has to be indicated for the conversion.
    :param three_dim: Boolean defining whether data contains three dimensional coordinates
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

        if three_dim:
            format_col = ['time', 'animal_id', 'x', 'y', 'z']
        else:
            format_col = ['time', 'animal_id', 'x', 'y']

        # check if all required columns are there in the right format
        if three_dim:
            if 'time' in data and 'animal_id' in data and 'x' in data and 'y' in data and 'z' in data:
                contains_columns = True
            else:
                contains_columns = False
        else:
            if 'time' in data and 'animal_id' in data and 'x' in data and 'y' in data:
                contains_columns = True
            else:
                contains_columns = False
        if contains_columns:
            # Check if 'time' attribute is integer-
            if is_numeric_dtype(data['time']):
                data.sort_values(['time', 'animal_id'],
                                 ascending=True,
                                 inplace=True)
                # Check if 'time' attribute is string-
            elif is_string_dtype(data['time']):
                if time_format == "undefined":
                    data['time'] = pd.to_datetime(data['time'])
                    warnings.warn('As time was converted and no format was given as parameter, please check if the time conversion is correct.')
                else:
                    data['time'] = pd.to_datetime(data['time'], format=time_format)
                data.sort_values(['time', 'animal_id'],
                                 ascending=True,
                                 inplace=True)
            # check if 'x' and 'y' and 'z' are numeric-
            if not is_numeric_dtype(data['x']):
                warnings.warn(
                    "'x' attribute has to be of numeric data type for most of the analyses to be performed"
                )
            if not is_numeric_dtype(data['y']):
                warnings.warn(
                    "'y' attribute has to be of numeric data type for most of the analyses to be performed"
                )

            if three_dim:
                if not is_numeric_dtype(data['z']):
                    warnings.warn(
                        "'z' attribute has to be of numeric data type for most of the analyses to be performed"
                    )

            # Check if 'heading_angle' attribute is given in CSV file-
            if 'heading_angle' in data and np.issubdtype(
                    data['heading_angle'].dtype, np.number):
                print(
                    "\n'heading_angle' attribute is found and will be processed\n"
                )

            # Check if time contains duplicates and if so execute warning
            grouped_df = data.groupby("animal_id")
            for aid in grouped_df.groups.keys():
                if(grouped_df.get_group(aid)["time"].duplicated().any()):
                    warnings.warn(
                        "For some animals there are duplicate values for time! This might indicate some error in the "
                        "data and can lead to incorrect analysis."
                    )
                    break
                    
            return data
        else:
            raise ValueError('Movekit requires columns to be named {} but was given {} instead'.format(format_col, data.columns))

    except FileNotFoundError:
        print("The file could not be found.\nPath given: {0}\n\n".format(
            path_to_file))


def read_data(path, sheet=0, time_format="undefined", three_dim=False):
    """
    Function to import data from 'csv', 'xlsx' and 'xls' files.
    :param path: Complete path/relative path to Excel file along with file name
    :param sheet: name of specific sheet given, by default first sheet of the excel workbook
    :param time_format: If time is given in an unusual format, the format has to be indicated for the conversion.
    :param three_dim: Boolean defining whether data contains three dimensional coordinates
    :return: Pandas DataFrame containing imported data.
    """
    # Call appropriate IO function based on file extension
    # Split string based on '.' (dot)-
    file_split = path.split(".")

    if file_split[-1] == 'csv':

        return parse_csv(path, time_format, three_dim)
    elif file_split[-1] == 'xlsx':
        return parse_excel(path, sheet, time_format, three_dim)
    elif file_split[-1] == 'xls':
        return parse_excel(path, sheet, time_format, three_dim)
    else:
        raise ValueError(f'File extension {file_split[-1]} can not be imported. Imported file has to be of type ".csv" or ".xlsx" or ".xls".')


def read_movebank(path_to_file, animal_id = 'individual-local-identifier'):
    """
    Function to import csv and excel files from the Movebank database.
    :param path_to_file: Complete path/relative path to file along with file name
    :param animal_id: Column name of the unique animal identifier (converted to be animal_id)
    return: Data frame in a format required for using the movekit package.
    """

    # check file extension and import data as pandas df
    file_split = path_to_file.split(".")
    if file_split[-1] == 'csv':
        data = pd.read_csv(path_to_file)
    elif file_split[-1] == 'xlsx':
        data = pd.read_excel(path_to_file)
    elif file_split[-1] == 'xls':
        data = pd.read_excel(path_to_file)
    else:
        raise ValueError(
            f'File extension {file_split[-1]} can not be imported. Imported file has to be of type ".csv" or ".xlsx" or ".xls".')

    # convert latitude and longitude  to xy coordinate system
    data = convert_latlon(data, latitude='location-lat', longitude='location-long',replace=False)

    # rename time, animal_id, x and y column and order them to beginning of df
    data.rename({'timestamp': 'time', animal_id: 'animal_id'}, axis=1, inplace=True)
    cols = data.columns.tolist()
    for i in ['time','animal_id','x','y']:
        cols.remove(i)
    cols = ['time','animal_id','x','y'] + cols
    data = data[cols]

    # order observations by  time
    data['time'] = pd.to_datetime(data['time'])
    data.sort_values(['time', 'animal_id'],
                     ascending=True,
                     inplace=True)
    data = data.reset_index(drop=True)
    return data


def read_with_geometry(path, animal_id="name", time="time", x="x", y="y",  z="z", coordinate_conversion=False, time_format="undefined", geopandas = True):
    """
    Function to import files containing both data and geometry (f.e. GeoPackage, (Geo)json, Shapefile and many more).
    :param path: Complete path/relative path to file along with file name
    :param animal_id: Key name of the unique animal identifier (as f.e. defined as property value in the geojson feature)
    :param time: Key name of time (as defined f.e. as property value in the geojson feature)
    :param x: Key name of x variable (as defined f.e. as property value in the geojson feature)
    :param y: Key name of y variable (as defined f.e. as property value in the geojson feature)
    :param z: Key name of z variable (as defined f.e. as property value in the geojson feature)
    :param coordinate_conversion: Boolean defining whether the x,y (and z) coordinates are stored in geometry object of imported data (f.e. as geometry value in geojson feature).
    Note that if coordinate_conversion=True function searches for point object in geometry column and converts coordinates to individual columns of returned data frame.
    (In case of multiple geometries for an observation, so called 'geometry collections', the first point object is converted.)
    :param time_format: If time is given in an unusual format, the format has to be indicated for the conversion.
    :param geopandas: Boolean defining whether the returned data frame is a geopandas data frame (containing geometry objects in column 'geometry') or a pandas data frame (not containing a 'geometry' column).
    return: Geopandas or pandas data frame in a format required for using the movekit package.
    """

    # check extension
    file_split = path.split(".")
    if file_split[-1] != 'json' and file_split[-1] != 'geojson' and file_split[-1] != 'gpkg' and file_split[-1] != 'shp':
        warnings.warn(
            f'This function might not be able to import {file_split[-1]} files. If the import id unsuccessful try file of type ".json", ".geojson", ".gpkg" or ".shp".')

    # extract and convert data
    data = gpd.read_file(path)

    if coordinate_conversion:

        # check first observation to see if type in geometry column is point
        if isinstance(data['geometry'][0], shapely.geometry.point.Point):
            # get coordinates from points
            x = []
            y = []
            if len(list(data["geometry"][0].coords)[0]) == 3:  # check for 3d
                contains_z = True
                z = []
                for i in range(len(data)):
                    x.append(data['geometry'][i].x)
                    y.append(data['geometry'][i].y)
                    z.append(data['geometry'][i].z)
            else:
                contains_z = False
                for i in range(len(data)):
                    x.append(data['geometry'][i].x)
                    y.append(data['geometry'][i].y)

        elif isinstance(data['geometry'][0], shapely.geometry.collection.GeometryCollection):
            for j in range(len(data['geometry'][0].geoms)): #if it is geometry collection, find position of the point
                if isinstance(data['geometry'][0].geoms[j], shapely.geometry.point.Point):
                    break
            if not isinstance(data['geometry'][0].geoms[j], shapely.geometry.point.Point):  # if no object in geometry collection is point
                warnings.warn('The geometry column of the given data does not contain any point object from which the coordinates could be extracted.'
                              'Thus, if data already contains columns except the geometry column with coordinates redefine coordinate_conversion to False,'
                              'otherwise the data can not be imported with this function.')
                return
            # get coordinates from points
            x = []
            y = []
            if len(list(data["geometry"][0].geoms[j].coords)[0]) == 3:  # check for 3d
                contains_z = True
                z = []
                for i in range(len(data)):
                    x.append(data['geometry'][i].geoms[j].x)
                    y.append(data['geometry'][i].geoms[j].y)
                    z.append(data['geometry'][i].geoms[j].z)
            else:
                contains_z = False
                for i in range(len(data)):
                    x.append(data['geometry'][i].geoms[j].x)
                    y.append(data['geometry'][i].geoms[j].y)
        else:  # geometry column has no point object
            warnings.warn(
                'The geometry column of the given data does not contain any point object from which the coordinates could be extracted.'
                'Thus, if data already contains columns except the geometry column with coordinates redefine coordinate_conversion to False,'
                'otherwise the data can not be imported with this function.')
            return
        data['x'] = x
        data['y'] = y
        if contains_z:
            data['z'] = z

        # rename columns
        data.rename(columns={f'{animal_id}': 'animal_id', f'{time}': 'time'}, inplace=True)

    else:
        if z in data.columns:
            data.rename(columns={f'{animal_id}': 'animal_id', f'{time}': 'time', f'{x}': 'x', f'{y}': 'y', f'{z}': 'z'},
                        inplace=True)
        else:
            data.rename(columns={f'{animal_id}': 'animal_id', f'{time}': 'time', f'{x}': 'x', f'{y}': 'y'},
                        inplace=True)


    # check if all required columns are there in the right format
    # Check if 'time' attribute is integer-
    if is_numeric_dtype(data['time']):
        data.sort_values(['time', 'animal_id'],
                         ascending=True,
                         inplace=True)
        # Check if 'time' attribute is string-
    elif is_string_dtype(data['time']):
        if time_format == "undefined":
            data['time'] = pd.to_datetime(data['time'])
            warnings.warn(
                'As time was converted and no format was given as parameter, please check if the time conversion is correct.')
        else:
            data['time'] = pd.to_datetime(data['time'], format=time_format)
        data.sort_values(['time', 'animal_id'],
                         ascending=True,
                         inplace=True)
    # check if 'x' and 'y' and 'z' are numeric-
    if not is_numeric_dtype(data['x']):
        warnings.warn(
            "'x' attribute has to be of numeric data type for most of the analyses to be performed"
        )
    if not is_numeric_dtype(data['y']):
        warnings.warn(
            "'y' attribute has to be of numeric data type for most of the analyses to be performed"
        )
    if 'z' in data.columns:
        if not is_numeric_dtype(data['z']):
            warnings.warn(
                "'z' attribute has to be of numeric data type for most of the analyses to be performed"
            )

    # Check if 'heading_angle' attribute is given in CSV file-
    if 'heading_angle' in data and np.issubdtype(
            data['heading_angle'].dtype, np.number):
        print(
            "\n'heading_angle' attribute is found and will be processed\n"
        )

    # Check if time contains duplicates and if so execute warning
    grouped_df = data.groupby("animal_id")
    for aid in grouped_df.groups.keys():
        if (grouped_df.get_group(aid)["time"].duplicated().any()):
            warnings.warn(
                "For some animals there are duplicate values for time! This might indicate some error in the "
                "data and can lead to incorrect analysis."
            )
            break

    if not geopandas:  # if returned data frame is desired to be pandas df and not geopandas df
        data = pd.DataFrame(data.drop(columns=['geometry']))

    return data.reset_index(drop=True)
