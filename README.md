# Deep Neural Networks Overview

<<TODO>> = need to add explanation from notes.

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

We initialise weights to be a random number because <<TODO>>

## Forward Propagation


## Calculate Cost


## Backward Propagation


## Update Parameters