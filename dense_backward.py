import numpy as np

def dense_backward(dZ, cache):
    """Dense (fully-connected) layer backpropagation."""
    A_prev, W, b = cache
    m = A_prev.shape[1] # no of datapoints
    dW = 1./m * A_prev.T.dot(dZ) # gradients w.r.t weights
    db = 1./m * np.sum(dZ, axis = 0, keepdims = True) # gradients w.r.t biases
    dA_prev = dZ.dot(W.T) # gradients w.r.t previous layer activations
    
    return dA_prev, dW, db
