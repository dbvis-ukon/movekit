import multiprocessing
import time

def wait(i):
    time.sleep(2)
    print("a")


if __name__ == "__main__":
    start_time = time.perf_counter()
    pool = multiprocessing.Pool()
    result = pool.map(wait, range(10))
    end_time = time.perf_counter()
    print(f'Executing in Pools: {end_time - start_time}')
