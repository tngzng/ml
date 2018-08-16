import numpy as np


class LinearRegression:
    def __init__(self):
        self.m = None
        self.b = None

    def train(self, X, Y):
        '''
        find the slope m and y-intercept b of the line that minimizes the
        square distance of each value x in the input X to the line defined by m and b

        to find m and b, we use the following formulas, based on the partial derivatives
        of m and b to the squared error:

        m = ((avg(X) * avg(Y)) - avg(X * Y)) / (avg(X)**2 - avg(X**2))

        b = avg(Y) - m * avg(X)

        for background on how these formulas are derived, khan academy has a great explainer:

        https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/squared-error-of-regression-line
        '''
        x_bar = X.mean()
        y_bar = Y.mean()
        xy_bar = (X * Y).mean()
        x_sq_bar = (X**2).mean()

        self.m = ((x_bar * y_bar) - xy_bar) / (x_bar**2 - x_sq_bar)
        self.b = y_bar - self.m * x_bar

        return self.m, self.b

    def predict(self, x):
        '''
        plug the input x into the formula for the line we've fit to the training data

        you've no doubt seen this formula before, but here it is *just* in case:

        y = mx + b
        '''
        predicted_y = self.m * x + b
        return predicted_y
