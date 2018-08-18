import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))
import unittest
import numpy as np

from regularization import l1, l2


class TestRegularization(unittest.TestCase):
    def setUp(self):
        # the sum of the weights of these two arrays are equal
        self.even_weights = np.array([100] * 100)
        self.shrunk_to_zero = np.array([10000] + [0] * 99)

    def test_l2(self):
        # l2 prefers even weights and assigns a higher penalty to shrunk_to_zero
        l2_even_weights = l2(self.even_weights)
        l2_shrunk_to_zero = l2(self.shrunk_to_zero)
        assert l2_shrunk_to_zero > l2_even_weights

    def test_l1(self):
        # l1 treats both the same because it looks at absolute value rather than squares
        l1_even_weights = l1(self.even_weights)
        l1_shrunk_to_zero = l1(self.shrunk_to_zero)
        assert l1_even_weights == l1_shrunk_to_zero


if __name__ == '__main__':
    unittest.main()
