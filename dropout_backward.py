import numpy as np

def dropout_backward(dA, dropout_mask, dropout_rate):
    """Dropout backpropagation which scales up un-dropped activation signals"""
    return dA * dropout_mask / (1.0 - dropout_rate)
