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
    all_normalized_features = []
    for features in all_features:
        feature_min = features.min()
        feature_max = features.max()
        # apply min max to the features vector
        normalized_features = (features - feature_min) / float(feature_max - feature_min)
        all_normalized_features.append(normalized_features)

    # transpose again to return the features to columns rather than rows
    normalized_X = np.array(all_normalized_features).T
    return normalized_X


feature_1_vals = np.array(range(10))
feature_2_vals = feature_1_vals * 2
X = np.array(zip(feature_1_vals, feature_2_vals))
print(min_max_scaler(X))