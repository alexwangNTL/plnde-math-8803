import numpy as np

def meshgrid_2d(x, y):
    n, m = len(x), len(y)
    x = np.reshape(x, (1, n))
    y = np.reshape(y, (m, 1))
    X = np.tile(x, (m, 1))
    Y = np.tile(y, (1, n))
    return X, Y

def meshgrid_3d(x, y, z):
    n, m, l = len(x), len(y), len(z)
    x = np.reshape(x, (1, 1, n))
    y = np.reshape(y, (1, m, 1))
    z = np.reshape(z, (l, 1, 1))
    X = np.tile(x, (l, m, 1))
    Y = np.tile(y, (l, 1, n))
    Z = np.tile(z, (1, m, n))
    return X, Y, Z
