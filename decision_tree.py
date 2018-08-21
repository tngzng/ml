import math

import numpy as np


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    def fit(self, X_train, y_train, feature_names):
        root_node = DecisionNode(X_train, y_train, feature_names)
        self.root = root_node
        self.branch(root_node)

    def predict(self, x):
        pass

    def calculate_entropy(self, classifications):
        '''
        :param np.array classifications: numpy array of True/False classifications for a possible branch
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

    def branch(self, decision_node):
        '''
        calculate all possible feature splits for the data in a decision node
        be dumb and just recalculate features we've already split on
        for the best split, assign the feature to the passed decision_node
        and create new children nodes
        recursively call branch on the children nodes if they're not pure within some threshold
        '''
        feature_sets = decision_node.data.T
        entropies = []
        all_classifications = []
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
            all_classifications.append(classifications_by_feature)

        info_gain = np.array([starting_entropy - entropy for entropy in entropies])
        highest_info_gain_i = info_gain.argmax()
        highest_info_feature = feature_names[highest_info_gain_i]
        # make new decision node for highest info gain feature
        # store all data on root decision node
        decision_node = DecisionNode(highest_info_feature, X_test)
        # make children for root for each feature value (ie sunny, rainy)
        highest_info_classifications = all_classifications[highest_info_gain_i]
        for classifications in highest_info_classifications:
            # we actually need to pass the data associated w the subset of X_test that matches a given feature attr
            # this is just the y labels currently
            child_node = DecisionNode(classifications)
            # recursively perform the same split for the child nodes
            # TODO add check to end when all leaves are pure OR max_tree_depth is reached
            self._branch(child_node)


class DecisionNode:
    def __init__(self, data, y_values, feature_names, feature=None, classification=None):
        self.data = data
        self.y_values = y_values
        self.feature_names = feature_names
        self.feature = feature
        self.children_by_attribute = {}
        self.classification = classification
    def add_child(self, classification, decision_node):
        self.children[classification] = decision_node


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
X_train = play_or_not[:, :-1]
y_train = (play_or_not[:, -1:]).reshape(len(play_or_not),)
y_train = np.array([True if x == 'True' else False for x in y_train])
feature_names = ['outlook', 'temp', 'humidity', 'windy']
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train, feature_names)