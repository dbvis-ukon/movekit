import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .feature_extraction import *


def plot_movement(data, frm, to):
    """
    Plot 'x' and 'y' attributes for given Pandas DataFrame in specified time frame.

    :param data: Pandas DataFrame (should be sorted by 'time' attribute).
    :param frm: Starting from time step.
    :param to: Ending to time step.
    :return: None.
    """

    plt.scatter(x='x',
                y='y',
                data=data.loc[(data['time'] >= frm) & (data['time'] <= to), :])
    plt.title("Plotting 'x' and 'y' coordinates")
    plt.xlabel("'x' coordinate")
    plt.ylabel("'y' coordinate")
    plt.grid()
    plt.show()

    return None


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

    :param avg_speed_data: dictionary containing grouping of data by 'animal_id' including average speed feature.
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

    :param data_animal_id_groups: DataFrame containing 'x' & 'y' attributes.
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
