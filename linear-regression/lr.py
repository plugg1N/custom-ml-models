import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class LinearRegression():
    def __init__(self, learning_rate = 1e-3, iterations = 500):
        self.learning_rate = learning_rate
        self.iterations = iterations

        
    def fit(self, X, Y):
        # num of examples, num of features
        self.m, self.n = np.array(X).shape
            

        # weights vector
        self.w = np.zeros(self.n)

        # bias
        self.b0 = 0

        # init training data
        self.X, self.Y = np.array(X), np.array(Y)

        # perform Gradient Descent
        for i in range(self.iterations):
            self.update_weights()

        return self


    def update_weights(self):
        # get predictions
        y_pred = self.predict(self.X)

        # calculate gradients
        dw = - (2 * self.X.T.dot(self.Y - y_pred)) / self.m
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

        elif metric == 'mse':
            for i in range(samples):
                sum_ += (Y_true[i] - Y_preds[i])**2

        return sum_ / samples
        



# Get data from csv
data = pd.read_csv('references/salary_data.csv')

X = data.iloc[:,:-1].values
y = data.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


learning_r = 1e-3
iterations = 10_000
    
lr = LinearRegression(iterations = iterations)
lr.fit(X_train, y_train)


print(lr.predict(X_test))
print(lr.score(y_test, lr.predict(X_test)))

