"""
This is a logistic regression model for only binary classification
date: 06.11.2023

Model works slower than expected, but has some potential after
some tweaks. This was a headache to make and thus, code is not
optimized and looks ugly. Looking forward to optimize it in future.

Test file of this model is in "test.py" file in "../" dir.
"""


import copy
import numpy as np

class logistic_regression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations


    def fit(self, x, y):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        
        for i in range( self.iterations ):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)


            
    def compute_loss(self, y_true, y_pred) -> float:
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + self.learning_rate)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + self.learning_rate)
        return -np.mean(y_zero_loss + y_one_loss)


    
    def compute_gradients(self, x, y_true, y_pred) -> [float, float]:
        # derivative of binary cross entropy
        diff =  y_pred - y_true

        # bias
        db = np.mean(diff)

        # weights
        dw = np.matmul(x.transpose(), diff)
        dw = np.array([np.mean(grad) for grad in dw])

        return dw, db


    
    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - self.learning_rate * error_w
        self.bias = self.bias - self.learning_rate * error_b


        
    def predict(self, x) -> list:
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]



    # calculate sigmoid
    def _sigmoid(self, x) -> float:
        return np.array([self._sigmoid_function(value) for value in x])
    
    def _sigmoid_function(self, x) -> np.array:
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        return np.exp(x) / (1 + np.exp(x))


    # transform values
    def _transform_x(self, x) -> list:
        return copy.deepcopy(x).values
    
    def _transform_y(self, y) -> list:
        return copy.deepcopy(y).values.reshape(y.shape[0], 1)
                        
