import numpy as np
from utils import sigmoid, sigmoid_backward, relu, relu_backward

np.random.seed(1)

def initialize_parameters(layer_dimensions):
    """
    Initialize weights and biases for each layer.
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dimensions)

    for l in range(1, L):
        parameters["W" + str(l)] = np.multiply(np.random.randn(layer_dimensions[l], layer_dimensions[l-1]), 0.01)
        parameters["b" + str(l)] = np.zeros((layer_dimensions[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dimensions[l], layer_dimensions[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dimensions[l], 1))

    return parameters


# Forward Propagation
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    linear_cache contains (A, W, b) for layer l.
    activation_cache contains the the (Z) for layer l.
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache) # [((A, W, b),(Z))]

    return A, cache


def forward_propagate(X, parameters):
    caches = []
    A = X

    # Since parameters contains W, b for each layer we divide by 2.
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)


    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, cache


# Cost
def compute_cost(AL, Y):
    m = Y.shape[1]
    
    cost = np.multiply((-1 / m), np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y))))
    
    # Ensure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost


# Back propagation
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.multiply(1/m, np.dot(dZ, A_prev.T))
    db = np.multiply(1/m, np.sum(dZ, axis=1, keepdims=True))
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def backward_propagate(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(0, L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - np.multiply(learning_rate, grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - np.multiply(learning_rate, grads["db" + str(l+1)])

    return parameters