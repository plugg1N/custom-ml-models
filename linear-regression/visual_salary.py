from linear_regression import linear_regression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# Prepare data
data = pd.read_csv('references/salary_data.csv')

# Reshape to fit the model (cause X.shape returns "(24, )" )
X = np.array(data['YearsExperience']).reshape(len(data), 1)
y = np.array(data['Salary']).reshape(len(data), 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)


# Train a model
model = linear_regression(iterations=5_000)
model.fit(X_train, y_train)

# Get predictions
y_preds = model.predict(X_test)

# Scatter dots
plt.scatter(X_test, y_test, color='blue')    # Test dots
plt.scatter(X_train, y_train, color='red')   # Train dots

plt.plot(X_test, y_preds, color='green')     # Our model


plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])


plt.savefig('salary_data_visualization.png')
print("Plot was saved successfully!")

