import math

import numpy as np


class DecisionTreeClassifier:
    pass


play_or_not = np.array([['sunny',    'hot',  'high',   False, False],
                        ['sunny',    'hot',  'high',   True,  False],
                        ['overcast', 'hot',  'high',   False, True],
                        ['rainy',    'mild', 'high',   False, True],
                        ['rainy',    'cool', 'normal', False, True],
                        ['rainy',    'cool', 'normal', True,  False],
                        ['overcast', 'cool', 'normal', True,  True],
                        ['sunny',    'mild', 'high',   False, False],
                        ['sunny',    'cool', 'normal', False, True],
                        ['rainy',    'mild', 'normal', False, True],
                        ['sunny',    'mild', 'normal', True,  True],
                        ['overcast', 'mild', 'high',   True,  True],
                        ['overcast', 'hot',  'normal', False, True],
                        ['rainy',    'mild', 'high',   True,  False]])

# entropy(p1, p2, ..., pn) = - p1 * log2(p1) - p2 * log2(p2) ... - pn * log2(pn)
# in our case we have two p values: True for play or False for no play, so we can simplify to
# entropy(p, q) = - p * log2(p) - q * log2(q)
# since there are only two possible values, p is equal to 1 - q
X_test = play_or_not[:, :-1]
y_test = (play_or_not[:, -1:]).reshape(len(play_or_not),)
y_test = np.array([True if x == 'True' else False for x in y_test])
feature_names = ['outlook', 'temp', 'humidity', 'windy']

def calculate_entropy(classifications):
    '''
    :param np.array classifications: numpy array of True/False classifications for a given slice
    '''
    ratio_positive = float(classifications.sum()) / len(classifications)  # ratio of elements classified as True
    ratio_negative = 1.0 - ratio_positive
    entropy_p = entropy_q = 0.0
    if ratio_positive != 0.0:
        entropy_p = - ratio_positive * math.log(ratio_positive, 2)
    if ratio_negative != 0.0:
        entropy_q = - ratio_negative * math.log(ratio_negative, 2)
    entropy = entropy_p + entropy_q
    return entropy

starting_entropy = calculate_entropy(y_test)

feature_sets = X_test.T
entropies = []
for feature_set in feature_sets:
    classifications_by_feature = {}
    # construct a data structure that looks like this
    # {'rainy': [True, True, False, True, False],
    # 'overcast': [True, True, True, True],
    # 'sunny': [False, False, False, True, True]}
    # where the key is the feature and the
    # value is a list of classifications (True or False) for that feature
    for i, feature_val in enumerate(feature_set):
        if not classifications_by_feature.get(feature_val):
            classifications_by_feature[feature_val] = []
        classifications_by_feature[feature_val].append(y_test[i])
    entropy_for_feature = 0
    for feature_val in classifications_by_feature:
        feature_val_classifications = np.array(classifications_by_feature[feature_val])
        feature_val_entropy = calculate_entropy(feature_val_classifications)
        ratio_feature_val = float(len(feature_val_classifications)) / len(y_test)
        entropy_for_feature += ratio_feature_val * feature_val_entropy
    entropies.append(entropy_for_feature)

info_gain = np.array([starting_entropy - entropy for entropy in entropies])
highest_info_gain_i = info_gain.argmax()

# make new decision node for highest info gain feature w
# store all data on root decision node
# make children for root for each feature value (ie sunny, rainy)
# recursively perform the same split for the child nodes
# end when all leaves are pure OR max_tree_depth is reached