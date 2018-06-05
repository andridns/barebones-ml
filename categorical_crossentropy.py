import numpy as np

def categorical_crossentropy(probs, y):
    correct_log_p = -np.log(probs[np.arange(y.shape[0]), y])
    return np.mean(correct_log_p)
