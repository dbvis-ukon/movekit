import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns


def plot_x_y_attributes(data, frm, to):
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
