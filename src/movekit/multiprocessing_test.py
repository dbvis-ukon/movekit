import multiprocessing
import numpy as np
import pandas as pd
import time

from src.movekit.feature_extraction import *

# function to be called from different processes in pool
def fun(df):
    df = extract_features(df)
    return df

# function which is called from external file
def multi_fun(data):
    # creating list of different animals df's to split them in pool
    start_time = time.perf_counter()
    data = grouping_data(data)
    df_list = []
    for aid in data.keys():
        df_list.append(data[aid])
    end_time = time.perf_counter()
    print(f'First grouping: {end_time-start_time}')

    # use multiprocessing to call extract_features for each animal with different process
    if __name__ == "src.movekit.multiprocessing_test":
        start_time = time.perf_counter()
        pool = multiprocessing.Pool()
        result = pool.map(fun, df_list)
        end_time = time.perf_counter()
        print(f'Executing in Pool: {end_time - start_time}')

    # regroup in one big data frame and return
    start_time = time.perf_counter()
    big_df = pd.DataFrame()
    for df in result:
        big_df = big_df.append(df, ignore_index=True)
    end_time = time.perf_counter()
    print(f' Regrouping after Pool: {end_time - start_time}')
    return big_df


