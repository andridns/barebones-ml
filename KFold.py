import numpy as np

def Kfold(X, n_splits, shuffle=False, random_state=42):
    idx = np.arange(X.shape[0])
    if shuffle:
        print(f'Shuffling with random seed: {random_state}')
        np.random.seed(random_state) # for reproducibility
        idx = np.random.permutation(X.shape[0]) # shuffling row indices
    len_test_fold = int(X.shape[0] / n_splits) # no of test datapoints per fold
    for i in range(n_splits):
        test_fold_idx = idx[np.arange(len_test_fold*i, len_test_fold*(i+1), 1)] # list of test fold indices
        train_fold_idx = np.setdiff1d(np.arange(X.shape[0]), test_fold_idx)
        
        yield train_fold_idx, test_fold_idx
