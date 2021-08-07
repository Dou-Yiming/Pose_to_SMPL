import numpy as np


def transform(arr: np.ndarray, rotate=[1.,-1.,-1.]):
    for i in range(arr.shape[0]):
        origin = arr[i][3].copy()
        for j in range(arr.shape[1]):
            arr[i][j] -= origin
            for k in range(3):
                arr[i][j][k] *= rotate[k]
        arr[i][3] = [0.0, 0.0, 0.0]
    print(arr[0])
    return arr
