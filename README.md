# Deep Neural Networks Overview

This readme is an overview of the deep neural network implemented by following [Andrew Ng's deep learning course](https://www.coursera.org/learn/neural-networks-deep-learning).

## Initialisation
Initialise weights and biases for each layer.

```
def initialize_parameters(layer_dimensions):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dimensions)

    for l in range(1, L):
        parameters["W" + str(l)] = np.multiply(np.random.randn(layer_dimensions[l], layer_dimensions[l-1]), 0.01)
        parameters["b" + str(l)] = np.zeros((layer_dimensions[l], 1))

    return parameters
```

We initialise weights to be a random number instead of 0 to prevent the slowdown of gradient descent.

## Forward Propagation

`linear_forward` provides us with Z.

```
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache
```

`linear_activation_forward` passes Z through the activation function giving us A.

```
def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache
```

`forward_propagate` performs the full forward propagation, going through each layer, 1 -> L.

```
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

    return AL, cache
```

## Calculate Cost


## Backward Propagation


## Update Parameters