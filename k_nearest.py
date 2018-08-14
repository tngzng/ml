import numpy as np


def k_nearest(X_train, y_train, x_test, n):
    '''
    :param (np array) X_train:
    a numpy array of numpy arrays, where each nested array represents the feature
    vector for a given sample in our training set.

    :param (np array) y_train: a numpy array, where each element represents the
    label corresponding to a given sample from X_train.

    :param (np array) x_test: a numpy array, representing the feature vector for
    a new sample we would like to predict.

    :param (int) n: the number of nearest neighbors to look at.

    :returns (str): the predicted label for x_test.
    '''
    if n % 2 != 1:
        raise TypeError('n must be an odd integer.')
    if n > len(X_train):
        raise TypeError('n must be smaller than the training set X_train.')
    # calculate euclidean distances between each element in X_train and x_test
    distances = np.array([np.linalg.norm(x_train - x_test) for x_train in X_train])

    # get the n nearest neighbors
    nearest_indices = distances[:n]
    nearest_neighbors = np.array([y_train[int(i)] for i in nearest_indices])

    # predict the label of x_test based on the majority label of its nearest neighbors
    labels, first_label_indices = np.unique(nearest_neighbors, return_inverse=True)
    counts = np.bincount(first_label_indices)
    majority_index = counts.argmax()
    predicted_label = labels[majority_index]
    return predicted_label
