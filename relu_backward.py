import numpy as np

def relu_backward(dA, cache):
    """ReLU backward propagation to return gradient w.r.t pre-activation matrix.""" 
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
