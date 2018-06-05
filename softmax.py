def softmax(Z):
    Z -= np.max(Z, axis=1, keepdims=True) # normalization trick for numerical stability
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
