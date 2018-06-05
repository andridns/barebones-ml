import numpy as np

def train_test_split(X, y, test_size, random_state=42):
    np.random.seed(random_state) # for reproducibility
    shuffled_idx = np.random.permutation(X.shape[0]) # shuffling row indices
    X, y = X[shuffled_idx], y[shuffled_idx] # reassign X and y with shuffled indices
    n_test = int(X.shape[0] * test_size) # no of test datapoints
    X_train, X_test = X[:-n_test], X[-n_test:] # slicing 
    y_train, y_test = y[:-n_test], y[-n_test:]

    return X_train, X_test, y_train, y_test
