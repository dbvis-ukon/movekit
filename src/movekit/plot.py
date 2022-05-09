import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from .feature_extraction import *
import seaborn as sns
import folium
from tqdm import tqdm
import warnings
import moviepy.editor as mp


def plot_movement(data, frm, to):
    """
    Plot 'x' and 'y' attributes for given Pandas DataFrame in specified time frame.

    :param data: Pandas DataFrame (should be sorted by 'time' attribute).
    :param frm: Starting from time step. Note that if time is stored as a date (if input data has time not stored as numeric type it is automatically converted to datetime) parameter has to be set using an datetime format: mkit.plot_movement(data, 2008-01-01, 2010-10-01)
    :param to: Ending to time step.
    :return: None.
    """

    sns.relplot(x='x',
                y='y',
                data=data.loc[(data['time'] >= frm) & (data['time'] <= to), :],
                hue='animal_id',
                palette='tab10')
    plt.title("Plotting 'x' and 'y' coordinates")
    plt.xlabel("'x' coordinate")
    plt.ylabel("'y' coordinate")
    plt.grid()
    plt.show()

    return None


def animate_movement(data, viewsize):
    """
    Animated version of plot_movement function.
    Animates 'x' and 'y' attributes for given Pandas DataFrame in specified time frame.
    :param data: Pandas DataFrame (should be sorted by 'time' attribute).
    :param viewsize: Int. Define how many time steps/frames should be visible in the animation.
    """
    
    xmin = data['x'].min()
    xmax = data['x'].max()
    ymin = data['y'].min()
    ymax = data['y'].max()

    fig = plt.figure()
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))

    scat = ax.scatter(x='x', y='y', data=data.loc[(data['time'] >= 0) & (data['time'] <= viewsize), :]) # first frame 
    
    def animate(i): # update frame sequentially
        subset = data.loc[(data['time'] >= i) & (data['time'] <= i+viewsize), :]
        x = subset['x']
        y = subset['y']
        scat.set_offsets(np.c_[x, y])   
    
    anim = animation.FuncAnimation(
        fig, animate, interval=100, frames=data['time'].max())
 
    plt.draw()
    plt.show()
    return anim # need to return anim otherwise it gets eaten by garbage collector and animation is not updated


def plot_animal_timesteps(data_animal_id_groups):
    """
    Plot the number of time steps for each 'animal_id'

    :param data_animal_id_groups: Dictionary containing grouping of data by 'animal_id'.
    :return: None
    """
    # Initialize Python 3.X dict to hold number of time steps
    # for each animals-
    animals_timesteps = {}

    for aid in data_animal_id_groups.keys():
        animals_timesteps[aid] = data_animal_id_groups[aid]['time'].count()

    # Sort 'animals_timesteps' dict in ascending order-
    # sorted((val, key) for (key, val) in animals_timesteps.items())
    # [(43201, 312), (43201, 511), (43201, 607), (43201, 811), (43201, 905)]

    plt.bar(range(len(animals_timesteps)),
            list(animals_timesteps.values()),
            align='center')

    plt.xticks(range(len(animals_timesteps)), list(animals_timesteps.keys()))
    plt.title("Barchart: number of timesteps versus animals")
    plt.xlabel("Animal IDs")
    plt.ylabel("Number of time steps")
    plt.yticks(rotation=20)

    plt.show()

    return None


def plot_pace(avg_speed_data, feature="speed"):
    """
    Plot average speed extracted feature for each animal.

    :param avg_speed_data: pandas Dataframe including average speed feature.
    :return: None.
    """

    # Group data into dictionary with animal_id as keys
    data_animal_id_groups = grouping_data(avg_speed_data)

    # Init dictionary to hold requested parameter
    animals_vals = {}

    if feature == "speed":

        # Fill dictionary with average speed feature
        for aid in data_animal_id_groups.keys():
            animals_vals[aid] = data_animal_id_groups[aid]['average_speed']
            animals_vals[aid] = np.asarray(animals_vals[aid])

        # List containing animal ids to be used within loop-
        animal_ids = list(animals_vals.keys())

        # Indices for stacking and making subplots in grid-
        c = 0

        fig, axs = plt.subplots(len(animal_ids),
                                sharex=True,
                                sharey=True,
                                figsize=(10, 10))

        fig.suptitle("Plot: Average Acceleration for each animal")

        while c < len(animal_ids):
            axs[c].plot(animals_vals[animal_ids[c]])
            axs[c].set_title("Animal ID: {0}".format(animal_ids[c]))
            c += 1

        for ax in axs.flat:
            ax.set(xlabel='number of time steps', ylabel='avg speed')

        # Hide x labels and tick labels for top plots and y ticks
        for ax in axs:
            ax.label_outer()

    if feature == "acceleration":

        # Fill dictionary with average acceleration feature
        for aid in data_animal_id_groups.keys():
            animals_vals[aid] = data_animal_id_groups[aid][
                'average_acceleration']
            animals_vals[aid] = np.asarray(animals_vals[aid])

        # List containing animal ids to be used within loop-
        animal_ids = list(animals_vals.keys())

        # Indices for stacking and making subplots in grid-
        c = 0

        fig, axs = plt.subplots(len(animal_ids),
                                sharex=True,
                                sharey=True,
                                figsize=(10, 10))

        fig.suptitle("Plot: Average Acceleration for each animal")

        while c < len(animal_ids):
            axs[c].plot(animals_vals[animal_ids[c]])
            axs[c].set_title("Animal ID: {0}".format(animal_ids[c]))
            c += 1

        for ax in axs.flat:
            ax.set(xlabel='number of time steps', ylabel='avg acc')

        # Hide x labels and tick labels for top plots and y ticks
        for ax in axs:
            ax.label_outer()

    plt.show()

    return None


def plot_animal(inp_data, animal_id):
    """
    Plot individual animal's 'x' and 'y' coordinates.

    :param inp_data: DataFrame containing 'x' & 'y' attributes.
    :param animal_id: ID of animal to be plotted.
    :return: None.
    """
    data_animal_id_groups = grouping_data(inp_data)
    if animal_id not in data_animal_id_groups.keys():
        print(
            "\nError!! animal id: {0} does NOT exist in the provided data!\n".
            format(animal_id))
        return None

    plt.plot(data_animal_id_groups[animal_id]['x'],
             data_animal_id_groups[animal_id]['y'])

    plt.title("Plotting animal id: {0}".format(animal_id))
    plt.xlabel("x coordinates")
    plt.ylabel("y coordinates")
    plt.show()

    return None



def plot_geodata(data, latitude_colname = "location-lat", longitude_colname = "location-long", animal_list=[], movement_lines=False):
    """
    Function to plot geo data on an interactive map using Open Street Maps.
    :param data: DataFrame containing the movement records
    :param latitude_colname: name of the column containing the latitude of each movement record
    :param longitude_colname: name of the column containing the longitude of each movement record
    :param animal_list: list containing animal_id's of all animals to be plotted (Default: every animal in data is plotted)
    :param movement_lines: Boolean whether movement lines between different location markers of animals are plotted
    return: map Object containing markers for each tracked animal position
    """
    warnings.warn("As plotting the geodata on an interactive map is  very time-consuming, it is recommended to reduce the size of the data as much as possible.")
    df = grouping_data(data)
    if animal_list != []:
        map = folium.Map(location=[df[animal_list[0]].loc[0, latitude_colname], data.loc[0, longitude_colname]],
                         zoom_start=15)  # start location is determined by first animal in specified animal_list
        for aid in tqdm(animal_list, position=0, desc="Plotting geo data for specified animals"):
            animal_data = df[aid]
            for i in range(len(animal_data)):
                folium.Marker(location=[animal_data.loc[i,latitude_colname], animal_data.loc[i,longitude_colname]],
                              tooltip=f'Animal ID: {animal_data.loc[i,"animal_id"]}',
                              popup=f'Animal ID: {animal_data.loc[i,"animal_id"]}\nTime: {animal_data.loc[i,"time"]}',
                              icon = folium.Icon(color="blue")).add_to(map)
                if movement_lines:
                    try:
                        folium.PolyLine(locations=[[animal_data.loc[i-1,latitude_colname], animal_data.loc[i-1,longitude_colname]],
                                               [animal_data.loc[i,latitude_colname], animal_data.loc[i,longitude_colname]]],
                                    tooltip=i,popup=f'Time Range of movement: From {animal_data.loc[i-1,"time"]} to {animal_data.loc[i,"time"]}').add_to(map)
                    except:
                        pass
    else:
        map = folium.Map(location=[data.loc[0, latitude_colname], data.loc[0, longitude_colname]],
                         zoom_start=15) # start location is determined by first observation
        for aid in tqdm(df.keys(),position=0, desc="Plotting geo data for all animals"):
            animal_data = df[aid]
            for i in range(len(animal_data)):
                folium.Marker(location=[animal_data.loc[i, latitude_colname], animal_data.loc[i, longitude_colname]],
                              tooltip=f'Animal ID: {animal_data.loc[i, "animal_id"]}',
                              popup=f'Animal ID: {animal_data.loc[i, "animal_id"]}\nTime: {animal_data.loc[i, "time"]}',
                              icon=folium.Icon(color="blue")).add_to(map)
                if movement_lines:
                    try:
                        folium.PolyLine(locations=[[animal_data.loc[i-1,latitude_colname], animal_data.loc[i-1,longitude_colname]],
                                               [animal_data.loc[i,latitude_colname], animal_data.loc[i,longitude_colname]]],
                                    tooltip=i,popup=f'Time Range of movement: From {animal_data.loc[i-1,"time"]} to {animal_data.loc[i,"time"]}').add_to(map)
                    except:
                        pass
    return map

def save_geodata_map(map, filename):
    """save the creates geodata map as a file
    :param map: map object to be saved.
    :param filename: name of the new created file containing the map.
    """
    try:
        map.save(filename)
    except:
        warnings.warn("Map could not be saved. Please try another file extension, f.e. '.html'")


def save_animation_plot(animation_object, filename):
    # save as gif
    writergif = animation.PillowWriter(fps=30)
    animation_object.save(f'{filename}.gif', writer=writergif)
    # save as mp4
    clip = mp.VideoFileClip(f'{filename}.gif')
    clip.write_videofile(f'{filename}.mp4')

