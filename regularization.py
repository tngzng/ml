import numpy as np


def l2(W, alpha=1.0):
    '''
    calculate the l2 penalty (used in ridge regression) for the input weights W

    this is for educational purposes to compare l1 and l2 penalty on different weights

    we expect to see l2 to produce a lower penalty when all values are minimized

    the l2 penalty formula is:

    alpha * sum(W**2)
    '''
    return alpha * (W**2).sum()


def l1(W, alpha=1.0):
    '''
    calculate the l1 penalty (used in lasso regression) for the input weights W

    this is for educational purposes to compare l1 and l2 penalty on different weights

    we expect to see l1 to produce a lower penalty when some values are zero

    the l1 penalty formula is:

    alpha * sum(abs(W))
    '''
    return alpha * abs(W).sum()
