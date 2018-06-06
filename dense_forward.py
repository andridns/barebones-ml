import numpy as np

def dense_forward(A, W, b):
    """Dense (fully-connected) layer forward propagation."""
    cache = (A, W, b)
    Z = A.dot(W) + b
    return Z, cache
