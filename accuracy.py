import numpy as np

def accuracy(preds, y):
    """Returns accuracy score given prediction probabilities and labels"""
    return np.sum(preds==y) / float(len(y))
