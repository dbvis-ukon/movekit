import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_movement(data, frm, to):
    """
	Plot 'x' and 'y' attributes for given Pandas
	DataFrame

	Input:
	data 	-	Pandas DataFrame (should be sorted by 'time' attribute)
	frm 	-	Starting from time step
	to 		-	Ending to time step

	Returns:
	Nothing
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
	Function to plot the number of time steps for each 'animal_id'

	Input:
	data_animal_id_groups	-	Python 3.X dictionary containing grouping of
								data by 'animal_id'

	Rerurns:
	Nothing
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


def plot_speed(avg_speed_data):
    """
	Function to plot average speed extracted feature for each animal

	Input:
	avg_speed_data	- Python 3.X dictionary containing grouping of data
						by 'animal_id' with average speed feature
						computed

	Returns:
	Nothing
	"""

    # A dictionary object to hold all groups obtained using group by-
    # Apply grouping using 'animal_id' attribute-
    data_animal_id = avg_speed_data.groupby('animal_id')

    # A dictionary object to hold all groups obtained using group by-
    data_animal_id_groups = {}

    # Get each animal_id's data from grouping performed-
    for animal_id in data_animal_id.groups.keys():
        data_animal_id_groups[animal_id] = data_animal_id.get_group(animal_id)

    # To reset index for each group-
    for animal_id in data_animal_id_groups.keys():
        data_animal_id_groups[animal_id].reset_index(drop=True, inplace=True)

    # Initialize Python 3.X dict to hold average speed
    # for each animals-
    animals_speed = {}

    for aid in data_animal_id_groups.keys():
        animals_speed[aid] = data_animal_id_groups[aid]['average_speed']
        animals_speed[aid] = np.asarray(animals_speed[aid])

    # n = len(animals_avg_speed)
    # n = n //2

    # List containing animal ids to be used within loop-
    animal_ids = list(animals_speed.keys())

    # Indices for stacking and making subplots in grid-
    c = 0
    i = 0
    j = 0

    # fig, axs = plt.subplots(n + 1, n, sharex = True)
    fig, axs = plt.subplots(len(animal_ids),
                            sharex=True,
                            sharey=True,
                            figsize=(10, 10))

    fig.suptitle("Plot: Average Speed for each animal")

    while c < len(animal_ids):
        axs[c].plot(animals_speed[animal_ids[c]])
        axs[c].set_title("Animal ID: {0}".format(animal_ids[c]))
        c += 1

    # fig.text(0.5, 0.04, "Average Speed", va = 'center')
    # fig.text(0.4, 0.5, "number of time steps", ha = 'center')

    # Set common labels-
    # ax.set_xlabel("number of time steps")
    # ax.set_ylabel("Average Speed")

    for ax in axs.flat:
        ax.set(xlabel='number of time steps', ylabel='avg speed')

    # Hide x labels and tick labels for top plots and y ticks for
    # right plots-
    # for ax in axs.flat:
    for ax in axs:
        ax.label_outer()

    plt.show()

    return None


def plot_acceleration(avg_acc_data):
    """
	Function to plot average acceleration extracted feature for each animal

	Input:
	avg_acc_data	- Pandas DataFrame containing relevant computed
						features

	Returns:
	Nothing
	"""

    # A dictionary object to hold all groups obtained using group by-
    # Apply grouping using 'animal_id' attribute-
    data_animal_id = avg_acc_data.groupby('animal_id')

    # A dictionary object to hold all groups obtained using group by-
    data_animal_id_groups = {}

    # Get each animal_id's data from grouping performed-
    for animal_id in data_animal_id.groups.keys():
        data_animal_id_groups[animal_id] = data_animal_id.get_group(animal_id)

    # To reset index for each group-
    for animal_id in data_animal_id_groups.keys():
        data_animal_id_groups[animal_id].reset_index(drop=True, inplace=True)

    # Initialize Python 3.X dict to hold average speed
    # for each animals-
    animals_acc = {}

    for aid in data_animal_id_groups.keys():
        animals_acc[aid] = data_animal_id_groups[aid]['average_acceleration']
        animals_acc[aid] = np.asarray(animals_acc[aid])

    # n = len(animals_avg_speed)
    # n = n //2

    # List containing animal ids to be used within loop-
    animal_ids = list(animals_acc.keys())

    # Indices for stacking and making subplots in grid-
    c = 0
    i = 0
    j = 0

    # fig, axs = plt.subplots(n + 1, n, sharex = True)
    fig, axs = plt.subplots(len(animal_ids),
                            sharex=True,
                            sharey=True,
                            figsize=(10, 10))

    fig.suptitle("Plot: Average Acceleration for each animal")

    while c < len(animal_ids):
        axs[c].plot(animals_acc[animal_ids[c]])
        axs[c].set_title("Animal ID: {0}".format(animal_ids[c]))
        c += 1

    # fig.text(0.5, 0.04, "Average Speed", va = 'center')
    # fig.text(0.4, 0.5, "number of time steps", ha = 'center')

    # Set common labels-
    # ax.set_xlabel("number of time steps")
    # ax.set_ylabel("Average Speed")

    for ax in axs.flat:
        ax.set(xlabel='number of time steps', ylabel='avg acc')

    # Hide x labels and tick labels for top plots and y ticks for
    # right plots-
    # for ax in axs.flat:
    for ax in axs:
        ax.label_outer()

    plt.show()

    return None


def plot_animal(data_animal_id_groups, animal_id):
    """
	Function to plot individual animal's 'x' and 'y' coordinates

	Input:
	data_animal_id_groups:		Python 3.X dictionary grouped by
								'animal_id' containing 'x' & 'y'
								attributes

	animal_id:					ID of animal to be plotted


	Returns:
	Nothing
	"""

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
