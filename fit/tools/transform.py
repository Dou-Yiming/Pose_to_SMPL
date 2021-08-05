import numpy as np


def transform(arr: np.ndarray):
    for i in range(arr.shape[0]):
        origin = arr[i][0].copy()
        for j in range(arr.shape[1]):
            arr[i][j] -= origin
            arr[i][j][1] *= -1
            arr[i][j][2] *= -1
        arr[i][0] = [0.0, 0.0, 0.0]
    return arr
