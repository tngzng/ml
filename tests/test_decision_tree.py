import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))
import unittest
import numpy as np

from decision_tree import DecisionTreeClassifier


class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        # our test data consists of weather features
        # and a boolean classification of whether or not we decide to play given each feature vector
        # test data taken from:
        # https://www.slideshare.net/marinasantini1/lecture-4-decision-trees-2-entropy-information-gain-gain-ratio-55241087
        self.feature_names =    ['outlook', 'temp',  'humidity', 'windy']
        play_or_not = np.array([['sunny',    'hot',  'high',     False,  False],
                                ['sunny',    'hot',  'high',     True,   False],
                                ['overcast', 'hot',  'high',     False,  True],
                                ['rainy',    'mild', 'high',     False,  True],
                                ['rainy',    'cool', 'normal',   False,  True],
                                ['rainy',    'cool', 'normal',   True,   False],
                                ['overcast', 'cool', 'normal',   True,   True],
                                ['sunny',    'mild', 'high',     False,  False],
                                ['sunny',    'cool', 'normal',   False,  True],
                                ['rainy',    'mild', 'normal',   False,  True],
                                ['sunny',    'mild', 'normal',   True,   True],
                                ['overcast', 'mild', 'high',     True,   True],
                                ['overcast', 'hot',  'normal',   False,  True],
                                ['rainy',    'mild', 'high',     True,   False]])
        self.X_train = play_or_not[:, :-1]
        y_train = (play_or_not[:, -1:]).reshape(len(play_or_not),)
        self.y_train = np.array([True if x == 'True' else False for x in y_train])

    def test_fit(self):
        clf = DecisionTreeClassifier()
        clf.fit(self.X_train, self.y_train, self.feature_names)
        # verify the decision tree looks like this
        #
        #                        feature:
        #                        outlook
        #                         / | \
        #                       /   |   \
        #             rainy   /  overcast \   sunny
        #                   /       |       \
        #                 /         |         \
        #            feature:     class:     feature:
        #            windy        True       humidity
        #            /   \                    /   \
        #   False  /       \  True     high /       \  normal
        #        /           \            /           \
        #      class:      class:      class:        class:
        #      True        False       False         True
        assert clf.root.feature == 'outlook'
        assert clf.root.children_by_attribute['rainy'].feature == 'windy'
        assert clf.root.children_by_attribute['overcast'].classification == True
        assert clf.root.children_by_attribute['sunny'].feature == 'humidity'
        # TODO child nodes should be their own vars for readability
        assert clf.root.children_by_attribute['rainy'].children_by_attribute['False'].classification == True
        assert clf.root.children_by_attribute['rainy'].children_by_attribute['True'].classification == False
        assert clf.root.children_by_attribute['sunny'].children_by_attribute['high'].classification == False
        assert clf.root.children_by_attribute['sunny'].children_by_attribute['normal'].classification == True

    def test_predict(self):
        clf = DecisionTreeClassifier()
        clf.fit(self.X_train, self.y_train, self.feature_names)
        x = np.array(['sunny', 'hot', 'high', False, False])
        clf.predict(x)
        # TODO test

    def test_calculate_entropy(self):
        clf = DecisionTreeClassifier()
        # TODO test


if __name__ == '__main__':
    unittest.main()
