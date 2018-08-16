import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))
import unittest
import numpy as np

from linear_regression import LinearRegression


class TestLinearRegressions(unittest.TestCase):
    def test_train_and_predict(self):
        # let's fit a line to the following x and y values
        #
        #   |
        # 3 |         x
        #   |
        # 2 |     x
        #   |
        # 1 | x
        #   |________________
        #     1   2   3   4
        X = np.array([1, 2, 3])
        Y = np.array([1, 2, 3])
        clf = LinearRegression()
        m, b = clf.train(X, Y)
        assert m == 1.0
        assert b == 0.0

        # now let's use our trained model to predict a y value for a new x
        predicted_y = clf. predict(4)
        assert predicted_y == 4.0


if __name__ == '__main__':
    unittest.main()
