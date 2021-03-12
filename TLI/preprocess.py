import numpy as np
from sklearn import preprocessing

def csv_to_list(path):
    result = []
    with open(path, "r") as f:
        for line in f:
            result.append(list(map(float, line.split(","))))
    return result

def del_last(arr, last_i=1):
    arr = np.array(arr)
    arr = arr.T[:-last_i]
    return arr.T

# Inserta valores en una matriz a partir de la
# posicion que se le indique.
def set_values_between(arr, index=1, values=[0]):
    arr = np.array(arr)
    arr = arr.T
    a = arr[:-index]
    b = arr[-index:]
    s = a.shape[1]
    matriz = []
    for val in values:
        matriz.append(np.full((s), val))
    matriz = np.array(matriz)
    return np.concatenate((a, matriz, b)).T

def split_dataset(X, Y, split=0.7):
    size = int(len(X)*split)
    X_train = X[:size]
    X_test = X[size:]

    Y_train = Y[:size]
    Y_test = Y[size:]

    return X_train, X_test, Y_train, Y_test


def load_simple(path):
    result = np.array(csv_to_list(path))
    r_t = result.T
    X = r_t[:-1].T

    # X = X[4:].T
    Y = r_t[-1]
    print("Max: {}".format(np.max(Y)))
    X = preprocessing.normalize(X, norm="max")
    Y = preprocessing.normalize([Y], norm="max")

    return split_dataset(X, Y[0])

def load_simple_no_angle(path):
    result = np.array(csv_to_list(path))
    r_t = result.T
    X = r_t[:-1]

    X = X[4:].T
    Y = r_t[-1]
    print("Max: {}".format(np.max(Y)))
    X = preprocessing.normalize(X, norm="max")
    Y = preprocessing.normalize([Y], norm="max")

    return split_dataset(X, Y[0])
