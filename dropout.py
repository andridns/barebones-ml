import numpy as np

def dropout(A, dropout_rate):
    """Dropout forward propagation with dropout mask returned."""
    dropout_mask = np.random.binomial([np.ones_like(A)], 1.0 - dropout_rate)[0]
    A *= dropout_mask / (1.0 - dropout_rate)
    return A, dropout_mask
