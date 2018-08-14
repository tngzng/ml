import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))
import unittest
import numpy as np

from k_nearest import k_nearest


class TestKNearest(unittest.TestCase):
    def test_k_nearest(self):
        # model the two dimensional feature space below as nested numpy arrays
        #
        #   |
        # 2 | o           x
        #   |
        # 1 | x   o       x
        #   |________________
        #     1   2   3   4

        X_train = np.array([[1, 1],  # x
                            [4, 1],  # x
                            [4, 2],  # x
                            [1, 2],  # o
                            [2, 1]]) # o
        y_train = np.array(['x', 'x', 'x', 'o', 'o'])
        x_test = X_train[0]
        test_vals = [
            (1, 'x'),  # when n is 1, the point at [1, 1] takes precedence
            (3, 'o'),  # when n is 3, the points at [1, 2] and [2, 1] take precedence
            (5, 'x'),  # when n is 5, all the points are taken into consideration
        ]
        for n, expected in test_vals:
            res = k_nearest(X_train, y_train, x_test, n)
            assert(res == expected)
