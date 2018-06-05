import numpy as np

def one_hot_encoding(v):
    return np.eye(len(np.unique(v)))[v]
