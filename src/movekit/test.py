from src.movekit.feature_extraction import *
from src.movekit.io import read_data
from src.movekit.multiprocessing_test import multi_fun
import time

# create random big data frame
data = read_data("examples/datasets/fish-5.csv")
df = data.copy()

for i in range(0, 140, 4):
    f = [[i]*1000 + [i+1]*1000 + [i+2]*1000 + [i+3]*1000 + [i+4]*1000]
    data2 = data.copy()
    data2['animal_id'] = f[0]
    df = df.append(data2, ignore_index=True)


# without multiprocessing
start_time = time.perf_counter()
extract_features(df)
finish_time = time.perf_counter()
print(finish_time - start_time)

# with multiprocessing
start_time = time.perf_counter()
multi_fun(df)
finish_time = time.perf_counter()
print(finish_time - start_time)
