import numpy as np

def sigmoid_backward(dA, cache):
    """Sigmoid backward propagation to return gradient w.r.t pre-activation matrix."""  
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ
