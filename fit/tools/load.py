import scipy.io
import numpy as np


def load(name, path):
    if name == 'UTD_MHAD':
        data = scipy.io.loadmat(path)
        arr = data['d_skel']
        new_arr = np.zeros([arr.shape[2], arr.shape[0], arr.shape[1]])
        for i in range(arr.shape[2]):
            for j in range(arr.shape[0]):
                for k in range(arr.shape[1]):
                    new_arr[i][j][k] = arr[j][k][i]
        return new_arr
    elif name == 'HumanAct12':
        return np.load(path)
