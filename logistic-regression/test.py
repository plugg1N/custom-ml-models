"""
Testing our model on breast
cancer dataset

I've worked on this model for
a very long time and I wanna relax, so I won't tune it
further, unforunately :(

Accuracy: 0.7808421052631579, with iter_am=3000
"""

from src.logistic_regression import logistic_regression
from load_data import X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score

lr = logistic_regression(learning_rate=1e-9, iterations=2_000)
lr.fit(X_train, y_train)

true = y_test.values.tolist()
preds = lr.predict(X_test)

print(accuracy_score(true, preds))

