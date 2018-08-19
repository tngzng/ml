import numpy as np


def accuracy(y_actual, y_predicted):
    '''
    the fraction of samples that were classified correctly
    '''
    pass


def recall(y_actual, y_predicted):
    '''
    true positive rate

    correctly predicted positive / all actual positive
    '''
    pass


def precision(y_actual, y_predicted):
    '''
    positive prediction rate

    correctly predicted positive / all predicted positive
    '''
    pass


def f_score(y_actual, y_predicted, beta=1.0):
    '''
    combine precision and recall into a single score, using beta to vary the weight of
    precision or recall

    a beta less than one will emphasize precision

    a beta greater than one will emphasize recall

    (1 + beta) * ((precision * recall) / ((beta * precision) + recall))
    '''
    pass
