import numpy as np


def accuracy_score(y_actual, y_predicted):
    '''
    the fraction of samples that were classified correctly
    '''
    is_true_positive_or_negative = y_actual == y_predicted
    true_positives_and_negatives = [x for x in is_true_positive_or_negative if x == True]
    return float(len(true_positives_and_negatives)) / len(y_actual)


def recall_score(y_actual, y_predicted):
    '''
    true positive rate

    correctly predicted positive / all actual positive
    '''
    is_true_positive = y_actual & y_predicted
    true_positives = [x for x in is_true_positive if x == True]
    actual_positives = [x for x in y_actual if x == True]
    return float(len(true_positives)) / len(actual_positives)


def precision_score(y_actual, y_predicted):
    '''
    positive predictive value

    correctly predicted positive / all predicted positive
    '''
    is_true_positive = y_actual & y_predicted
    true_positives = [x for x in is_true_positive if x == True]
    predicted_positive = [x for x in y_predicted if x == True]
    return float(len(true_positives)) / len(predicted_positive)


def f_score(y_actual, y_predicted, beta=1.0):
    '''
    combine precision and recall into a single score, using beta to vary the weight of
    precision or recall

    a beta less than one will emphasize precision

    a beta greater than one will emphasize recall

    (1 + beta) * ((precision * recall) / ((beta * precision) + recall))
    '''
    pass
