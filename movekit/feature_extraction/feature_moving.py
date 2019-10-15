

import pandas as pd
import numpy as np
import math


def grouping_data(processed_data):
    '''
    A function to group all values for each 'animal_id'

    Input: 'processed_data' which is processed Pandas DataFrame
    Returns: Python 3 dictionary where-
    key is animal_id, value in Pandas DataFrame for that 'animal_id'
    '''

    # A dictionary object to hold all groups obtained using group by-
    data_animal_id_groups = {}

    # Group by using 'animal_id' attribute-
    data_animal_id = processed_data.groupby('animal_id')

    # Get each animal_id's data from grouping performed-
    for animal_id in data_animal_id.groups.keys():
        data_animal_id_groups[animal_id] = data_animal_id.get_group(animal_id)

    # To reset index for each group-
    for animal_id in data_animal_id_groups.keys():
        data_animal_id_groups[animal_id].reset_index(drop = True, inplace = True)

    # Add additional attributes/columns to each groups-
    for aid in data_animal_id_groups.keys():
        data = [0 for x in range(data_animal_id_groups[aid].shape[0])]
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(Distance = data)
        data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(Average_Speed = data)
        # data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(Average_Acceleration = data)
        # data_animal_id_groups[aid] = data_animal_id_groups[aid].assign(Direction = data)

    return data_animal_id_groups


def compute_distance(data_animal_id_groups):
    '''
    Function to calculate the metric distance between two consecutive time frames/time stamps
    for each moving entity (in this case, fish)

    Input: Python 3 dictionary containing as key 'animal_id' & as value, Pandas DataFrame related to it
    Returns: Python 3 dictionary as input containing computation for distance
	'''

    # start_time = time.time()

    for aid in data_animal_id_groups.keys():
        print("\nComputing Distance for Animal ID = {0}\n".format(aid))

        for i in range(1, data_animal_id_groups[aid].shape[0] - 1):
            x1 = data_animal_id_groups[aid].iloc[i, 2]
            y1 = data_animal_id_groups[aid].iloc[i, 3]
            x2 = data_animal_id_groups[aid].iloc[i + 1, 2]
            y2 = data_animal_id_groups[aid].iloc[i + 1, 3]

            # Compute distance between 2 points-
            distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))

            """
            # Compute the direction in DEGREES-
            direction = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
            if matsh.isnan(direction):
                data_animal_id_groups[aid].loc[i, 'Direction'] = 0
            else:
                data_animal_id_groups[aid].loc[i, 'Direction'] = direction
            """

            # Insert computed distance to column/attribute 'Distance'-
            # animal_id.loc[i, 'Distance'] = distance
            data_animal_id_groups[aid].loc[i, 'Distance'] = distance

        # end_time = time.time()
        # print("\nTime taken to create distance & direction data = {0:.4f} seconds\n\n".format(end_time - start_time))
        # Time taken to create distance & direction data = 1013.1692 seconds

    return data_animal_id_groups


def compute_average_speed(data_animal_id_groups, fps):
    '''
    A function to compute average speed of an animal based on fps
    (frames per second) parameter. Calculate the average speed of a mover,
    based on the pandas dataframe and a frames per second (fps) parameter

    Formula used-
    Average Speed = Total Distance Travelled / Total Time taken

    Input:
            1.) Python 3 dictionary containing as key 'animal_id' & as value, Pandas DataFrame associated with it
            2.) 'fps' specifying frames per second parameter according to which average speed will be computed

    Output: Processed Python 3 dictionary passed to function containing computed average speed
    '''

    # start_time = time.time()

    for aid in data_animal_id_groups.keys():
        print("\nComputing Average Speed for Animal ID = {0}\n".format(aid))

        # for i in range (1, animal_id.shape[0] - fps + 1):
        for i in range(1, data_animal_id_groups[aid].shape[0] - fps + 1):
            tot_dist = 0	# total distance travelled

            for j in range(i, i + fps):
                tot_dist += data_animal_id_groups[aid].loc[j, "Distance"]
                data_animal_id_groups[aid].loc[i, "Average_Speed"] = (tot_dist / fps)

    # end_time = time.time()
    # print("\nTime taken to create Average Speed data = {0:.4f} seconds.\n".format(end_time - start_time))

    return data_animal_id_groups


def computing_stops(data_grouped, threshold_speed = 0.5):
    '''
    This function creates a new column called 'Stopped' where the value is 'yes'
    if 'Average_Speed' <= threshold_speed and 'no' otherwise
    '''
    for aid in data_grouped.keys():
        print("\nComputing stops for Animal ID = {0}\n".format(aid))

        # for i in range(1, data_animal_id_groups[aid].shape[0] - 1):
        data_grouped[aid]['Stopped'] = np.where(data_grouped[aid]['Average_Speed'] <= threshold_speed, 'yes', 'no')

    
    return data_grouped


    print("\nNumber of fishes stopped according to threshold speed = {0} is {1}".format(threshold_speed, data['Stopped'].eq('yes').sum()))
    print("Number of fishes moving according to threshold speed = {0} is {1}\n".format(threshold_speed, data['Stopped'].eq('no').sum()))


def feature_moving_summary(data_grouped_stops):
    '''
    Function to compute how long/time steps, each animal (animal_id)
    is in motion and is stationary
    This is done for each 'animal_id'

    Input: Python 3 dictionary containing 'Stopped' attribute from using 'computing_stops()' function
    Returns: Textual description per animal_id of the number of time steps for which they were
    in motion and were stationary
    '''

    for aid in data_grouped_stops.keys():
        print("\nanimal_id = {0} is in motion for = {1} time steps".format(aid, data_grouped_stops[aid]['Stopped'].eq('yes').sum()))
        print("animal_id = {0} is stationary for = {1} time steps\n".format(aid, data_grouped_stops[aid]['Stopped'].eq('no').sum()))

    return None


'''
# Example usage-
# Read in CSV file-
data = pd.read_csv("/path_to_file/fish-5.csv")

# Group data according to 'animal_id'-
grouped_data = grouping_data(data)

# Compute distance-
distance_data = compute_distance(grouped_data)

# Compute average speed-
average_speed = compute_average_speed(distance_data, fps = 3)

# Compute stops-
stops_data = computing_stops(average_speed, threshold_speed= 0.9)

# Get animal moving and stop summary-
feature_moving_summary(stops_data)
'''

