import numpy as np

def softmax_backward(probs, y):
    """Softmax backward propagation to return gradient w.r.t pre-activation matrix."""
    probs[np.arange(y.shape[0]), y] -= 1
    return probs
