import numpy as np


def l2(W, alpha=1.0):
    '''
    calculate the l2 penalty for the weights W

    where l2 penalty is equal to:

    alpha * sum(W**2)
    '''
    return alpha * (W**2).sum()


print(l2(np.array([1, 1, 2]), 2.0))
