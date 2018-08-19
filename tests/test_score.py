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

    def test_accuracy(self):
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


if __name__ == '__main__':
    unittest.main()
