import pandas as pd
import numpy as np


def presence_3d(data):
    return 'z' in data.columns


def angle(vec1, vec2):
    cos = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # circumvent float precision errors. limit to [-1,1] with this check
    if cos > 1:
        cos = 1
    elif cos < -1:
        cos = -1
    return np.arccos(cos)