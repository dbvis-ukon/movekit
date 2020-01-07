

"""
Example on using 'feature_extraction_combined.py' source file
"""

# %run ./feature_extraction_combined.py

data = pd.read_csv("/home/arjun/University_of_Konstanz/Hiwi/Work/Movement_Patterns/fish-5.csv")

data_grouped = grouping_data(data)

data_dist_direction = compute_distance_and_direction(data_grouped)

data_avg_speed = compute_average_speed(data_dist_direction, fps = 10)

data_avg_acc = compute_average_acceleration(data_avg_speed, fps=10)

