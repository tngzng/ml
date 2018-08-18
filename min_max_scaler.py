import numpy as np


def min_max_scaler(X):
    '''
    for each feature x in X, normalize the data points to fit between 0 and 1, using this formula:

    (x - min(X)) / (max(X) - min(X))

    intuitively, we see that the min feature will be assigned a new value of 0 because
    when x is the min feature the numerator of the expression will be 0. and when x
    is the max value for a feature it will be replaced with 1 because the numerator and
    the denominator will be the same.

    :param (np array) X: a numpy array of numpy arrays, where each nested
    array represents a feature vector.

    :returns (np array): a numpy array of numpy arrays, where each nested array
    of feature vectors has been scaled to a number between 0 and 1.
    '''
    # take the transpose of X to get each feature column as its own row
    all_features = X.T
    all_scaled_features = []
    for features in all_features:
        feature_min = features.min()
        feature_max = features.max()
        # apply min max to the features vector
        scaled_features = (features - feature_min) / float(feature_max - feature_min)
        all_scaled_features.append(scaled_features)

    # transpose again to return the features to columns rather than rows
    scaled_X = np.array(all_scaled_features).T
    return scaled_X
