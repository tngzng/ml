import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))
import unittest
import numpy as np

from score import accuracy_score, recall_score, precision_score


class TestScore(unittest.TestCase):
    def setUp(self):
        self.all_trues = np.array([True] * 10)
        self.half_true_half_false = np.array([False] * 5 + [True] * 5)

    def test_accuracy_score(self):
        # when half of our predictions are false negatives and half are true positives
        # our accuracy is 50%
        y_actual = self.all_trues
        y_predicted = self.half_true_half_false
        accuracy = accuracy_score(y_actual, y_predicted)
        assert accuracy == .5

        # when half of our predictions are false positives and half are true positives
        # our accuracy is still 50%
        y_actual = self.half_true_half_false
        y_predicted = self.all_trues
        accuracy = accuracy_score(y_actual, y_predicted)
        assert accuracy == .5

    def test_recall_score(self):
        # we get penalized for calling something false that's actually true
        # (ie penalized for false negatives)
        y_actual = self.all_trues
        y_predicted = self.half_true_half_false
        recall = recall_score(y_actual, y_predicted)
        assert recall == .5

        # we don't get penalized for calling something true that's actually false
        # (ie not penalized for false positives)
        y_actual = self.half_true_half_false
        y_predicted = self.all_trues
        recall = recall_score(y_actual, y_predicted)
        assert recall == 1.0

    def test_precision_score(self):
        # we don't get penalized for calling something false that's actually true
        # (ie not penalized for false negatives)
        y_actual = self.all_trues
        y_predicted = self.half_true_half_false
        precision = precision_score(y_actual, y_predicted)
        assert precision == 1.0

        # we get penalized for calling something true that's actually false
        # (ie penalized for false positives)
        y_actual = self.half_true_half_false
        y_predicted = self.all_trues
        precision = precision_score(y_actual, y_predicted)
        assert precision == .5


if __name__ == '__main__':
    unittest.main()
