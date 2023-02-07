import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from .feature_extraction import *
import seaborn as sns
import folium
from tqdm import tqdm
import warnings
# import moviepy.editor as mp
from pandas.api.types import is_numeric_dtype, is_string_dtype



def plot_movement(data, frm, to):
    """
    Plot 'x' and 'y' attributes for given Pandas DataFrame in specified time frame.
    :param data: Pandas DataFrame (should be sorted by 'time' attribute).
    :param frm: Starting from time step. Note that if time is stored as a date (if input data has time not stored as numeric type it is automatically converted to datetime) parameter has to be set using an datetime format: mkit.plot_movement(data, "2008-01-01", "2010-10-01")
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

    # check if time format has to be converted
    if not is_numeric_dtype(data['time']):
        time_values = np.array(data['time'])
        time_values = np.unique(time_values)
        indices = np.sort(time_values)
        converter = {}
        for i in range(len(time_values)):
            converter[indices[i]] = i
        data = data.replace({'time': converter})

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


def plot_animal_timesteps(data):
    """
    Plot the number of time steps for each 'animal_id'
    :param data_animal_id_groups: DataFrame containing movement records.
    :return: None
    """
    # Initialize Python 3.X dict to hold number of time steps
    # for each animals-
    animals_timesteps = {}

    data_animal_id_groups = grouping_data(data)
    for aid in data_animal_id_groups.keys():
        animals_timesteps[aid] = data_animal_id_groups[aid]['time'].count()

    # Sort 'animals_timesteps' dict in ascending order-
    # sorted((val, key) for (key, val) in animals_timesteps.items())
    # [(43201, 312), (43201, 511), (43201, 607), (43201, 811), (43201, 905)]

    sns.barplot(x=list(animals_timesteps.keys()), y=list(animals_timesteps.values()))

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

    if feature == "speed":

        g = sns.FacetGrid(avg_speed_data, row="animal_id")
        g.map(sns.lineplot, "time", "average_speed", alpha=.7)
        g.add_legend()
        g.set_titles("Average speed for each animal")

        for x in g.axes_dict.keys():
            g.axes_dict[x].set_title(f"Animal ID: {x}")
            g.axes_dict[x].set(xlabel='number of time steps', ylabel='avg speed')

        plt.show()

    if feature == "acceleration":

        g = sns.FacetGrid(avg_speed_data, row="animal_id")
        g.map(sns.lineplot, "time", "average_acceleration", alpha=.7)
        g.add_legend()
        g.set_titles("Average acceleration for each animal")

        for x in g.axes_dict.keys():
            g.axes_dict[x].set_title(f"Animal ID: {x}")
            g.axes_dict[x].set(xlabel='number of time steps', ylabel='avg acc')

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

    sns.relplot(x=data_animal_id_groups[animal_id]['x'],
             y=data_animal_id_groups[animal_id]['y'])

    plt.title("Plotting animal id: {0}".format(animal_id), y=0.985)
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
    """
    save the created geodata map as a file
    :param map: map object to be saved.
    :param filename: name of the new created file containing the map.
    """
    try:
        map.save(filename)
    except:
        warnings.warn("Map could not be saved. Please try another file extension, f.e. '.html'")


def save_animation_plot(animation_object, filename):
    """
    save animation as gif in working directory. (mp4 file is not working at the moment as moviepy import error)
    :param animation_object: created animation object
    :param filename: name of the two files which are created
    """
    # save as gif
    writergif = animation.PillowWriter(fps=30)
    animation_object.save(f'{filename}.gif', writer=writergif)
    """
    # save as mp4
    clip = mp.VideoFileClip(f'{filename}.gif')
    clip.write_videofile(f'{filename}.mp4')
    """


def plot_heatmap(data, time0_start, time0_end, round_digits=1, font_size=10, linewidth=0.5):
    """
    Plot a heatmap for the mover for user defined time interval.
    :param data: data frame returned by function getis_ord(): Data frame containing xy- interval coordinates and respective Getis-Ord statistic.
    :param time0_start: beginning time of the earliest interval included in the heatmap.
    :param time0_end: beginning time of the latest interval included in the heatmap.
    :param round_digits: for clear axis description the xy-values of the displayed intervals are rounded to have user defined number of digits.
    :param font_size: for clear axis description font size of the axis ticks can be defined.
    :param linewidth: width of the line dividing each cell in heatmap.
    """
    pd.options.mode.chained_assignment = None  # disable warning that we set values on slice of original dataframe
    data = data.loc[(data['time0'] >= time0_start) & (data['time0'] <= time0_end), :]
    data.loc[:, ['x0', 'x1', 'y0', 'y1']] = data[['x0', 'x1', 'y0', 'y1']].apply(lambda x: round(x, round_digits))
    data = pd.pivot_table(data, values='Getis-Ord Score', index=['y0', 'y1'],
               columns=['x0', 'x1'], aggfunc=np.mean)
    ax = sns.heatmap(data, linewidth=linewidth, cmap='Reds')
    ax.tick_params(labelsize=font_size)
    plt.show()
