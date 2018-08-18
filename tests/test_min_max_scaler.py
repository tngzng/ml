import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '../..'))
import unittest
import numpy as np

from min_max_scaler import min_max_scaler


class TestMinMaxScaler(unittest.TestCase):
    def test_min_max_scaler(self):
        # both features consist of five evenly spaced values
        feature_1_vals = np.array(range(5))
        feature_2_vals = feature_1_vals * 2
        X = np.array(zip(feature_1_vals, feature_2_vals))
        scaled_X = min_max_scaler(X)
        scaled_feature_1 = scaled_X.T[0]
        scaled_feature_2 = scaled_X.T[1]
        # both scaled features should now be five evenly spaced values between 0 and 1
        self.assertListEqual(list(scaled_feature_1), [0, .25, .5, .75, 1])
        self.assertListEqual(list(scaled_feature_2), [0, .25, .5, .75, 1])

if __name__ == '__main__':
    unittest.main()
