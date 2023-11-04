"""
This is a whole module for LinearRegression model

This version supports:

- Gradient Descent Tuning
- Learning rate and iterations_amount tuning
- All benefits of an OOP module
- Get predictions of a certain subset
- Get scores based on 3 supported metrics:
    - mae
    - mse
    - rmse

Everything was written manually with python libs,
except:
- numpy (lin. algebra)

GitHub: plugg1N

"""

import math
import numpy as np


class linear_regression():
    def __init__(self, learning_rate = 1e-3, iterations = 3000):
        self.learning_rate = learning_rate
        self.iterations = iterations


    def fit(self, X, Y):
        # num of examples, num of features
        self.m, self.n = X.shape


        # weights vector
        self.w = np.zeros(self.n)

        # bias
        self.b0 = 0

        # init training data
        self.X = X
        self.Y = Y

        # perform Gradient Descent
        for i in range(self.iterations):
            self._update_weights()

        return self


    def _update_weights(self):
        # get predictions
        y_pred = self.predict(self.X)

        # calculate gradients
        dw = - (2 * (self.X.T).dot(self.Y - y_pred)) / self.m
        db = - 2 * np.sum(self.Y - y_pred) / self.m

        # update weights
        self.w = self.w - self.learning_rate * dw
        self.b0 = self.b0 - self.learning_rate * db

        # update variables
        return self


    # linear function
    def predict(self, X):
        return X.dot(self.w) + self.b0


    # score model on metrics
    def score(self, Y_true, Y_preds, metric = 'mae'):
        sum_ = 0
        samples = Y_true.shape[0]

        # calc 'mae'
        if metric == 'mae':
            for i in range(samples):
                sum_ += abs(Y_true[i] - Y_preds[i])

        # calc 'mse'
        elif metric == 'mse':
            for i in range(samples):
                sum_ += (Y_true[i] - Y_preds[i])**2

        # calc 'rmse'
        elif metric == 'rmse':
            for i in range(samples):
                sum_ += (Y_true[i] - Y_preds[i])**2
            return math.sqrt(sum_ / samples)

        return sum_ / samples

