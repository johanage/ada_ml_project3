# defining the cost functions so that they work with autograd
# and can be optimized wrt the output activations
# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    return onehot_vector

def probs_to_binary(probabilities):
    return (np.round(probabilities)).astype(int)

def accuracy(y, a):
    return sum( [tuple(y[i]) == tuple(a[i]) for i in range(y.shape[0])] )/y.shape[0]

