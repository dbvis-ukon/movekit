"""
  Plot missing values 
  Author: Arjun Majumdar, Eren Cakmak
  Created: September, 2019
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_missing_values(data, animal_id):
    """
    Plot the missing values of an animal id against the time

    Input:
    'data' is the Pandas Data Frame containing CSV file
    'animal_id' is the ID of the animal who's missing values have to be plotted
    against the time

    Returns:
    Nothing
    """

    missing_time = data[data['time'].isnull()].index.tolist()
    missing_x = data[data['x'].isnull()].index.tolist()
    missing_y = data[data['y'].isnull()].index.tolist()

    # This visualizes the location(s) of missing values-
    # sns.heatmap(df.isnull(), cbar=False)
    # sns.heatmap(df.isnull())

    # Visualizing the count of missing values for all attributes-
    data.isnull().sum().plot(kind='bar')
    plt.xticks(rotation=20)
    plt.title("Visualizing count of missing values for all attributes")
    plt.show()
