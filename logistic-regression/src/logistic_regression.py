# imports
import math
import numpy as np
from sklearn.metrics import accuracy_score


# main class
class logistic_regression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations


    def fit(self, X, y):
        # m - samples, n - features
        self.m, self.n = X.shape

        self.w = np.zeros( self.n )
        self.b = 0

        for i in range(self.iterations):
            dot_product = np.matmul(self.w, X.T + self.b)
            pred = self._sigmoid(dot_product)
            loss = self.loss(y, y_pred)

            er_w, er_b = self._gradients(X, y, pred)

            self._update_weights(er_w, er_b)


    def _gradients(self, X, y_true, y_pred) -> [float, float]:
        diff = y_pred, y_true

        # Gradients for weights
        dw = np.matmul(X.T, diff)
        dw = np.array([np.mean(grad) for grad in dw])

        # Gradient for bias
        db = np.array(diff)

        return dw, db


    # Make predictions
    def predict(self, X) -> list:
        dot_product = np.matmul(X, self.w.T + self.b)
        probs = self._sigmoid(dot_product)

        return [1 if p > 0.5 else 0 for p in probs]
    

    # Update weights and bias
    def _update_weights(self, er_w, er_b):
        self.w = self.w - self.learning_rate * er_w
        self.b = self.b - self.learning_rate * er_b

        return self

        
    # Loss function (binary crossentropy)
    def loss(self, y_true, y_pred) -> float:
        zero_L = y_true * np.log(y_pred + self.learning_rate)
        one_L = (1 - y_true) * np.log(1 - y_pred + self.learning_rate)
        return -np.mean(zero_L + one_L)

        
    # Calculate sigmoid
    def _sigmoid_func(self, X) -> float:
        if X >= 0:
            return 1 / (1 + np.exp(-X))

        return np.exp(X) / (1 + np.exp(X))

    
    def _sigmoid(self, X) -> np.array:
        return np.array([self._sigmoid_func(val) for val in X])

    def score(y_true, y_pred) -> float:
        return accuracy_score(y_true, y_pred)
