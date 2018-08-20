import math

import numpy as np


class DecisionTree:
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
ratio_positive = float(y_test.sum()) / len(y_test)  # ratio of elements classified as True
ratio_negative = 1 - ratio_positive
starting_entropy = - ratio_positive * math.log(ratio_positive, 2) - ratio_negative * math.log(ratio_negative, 2)