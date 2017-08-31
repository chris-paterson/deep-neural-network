import numpy as np

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def sigmoid_backwards(dA, cache):
    Z = cache
    A = sigmoid(Z)
    dZ = dA * A * (1 - A)

    assert (dZ.shape == Z.shape)
    return dZ

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z

    assert(A.shape == Z.shape)
    return A, cache

def relu_backwards(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ
