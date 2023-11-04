"""
Simplest linear regression model without:

- Gradient Descent
- Multi-layer input


Sometimes cost can go up, due to the
fact that weights and bias become bigger than needed

Best params:
cost = 0.00503168492059078, w = 1.97851323163009, b = 0.13165922797451451

"""



import random
import numpy as np

# Data
X = [i for i in range(10)]
y = [i*2 for i in X]

# Constants
b0 = random.randint(1, 20)  # bias
alpha = 1e-3                # learning rate
eps = 1e-3                  # epsilon
w = random.randint(1, 20)   # random weight



# Calculate predictions based on Linear Regression method
def h(w, b0) -> list:
    return np.array(X).dot(w) + b0


# Calculate MSE
def cost(w, b0) -> float:
    result = 0

    for i in range(len(X)):
        result += (y[i] - h(w, b0)[i]) ** 2

    return result / len(X)


# Tuning (!NOT GRADIENT DESCENT)
for i in range(7_000):

    # Cost without tuning
    c = cost(w, b0)

    # Find derivatives for each weights and biases
    dw = (cost(w + eps, b0) - c) / eps
    db = (cost(w, b0 + eps) - c) / eps

    # Tune parameters
    w -= dw * alpha
    b0 -= db * alpha

    # Calculate cost and print
    print(f"cost = {cost(w, b0)}, w = {w}, b = {b0}")


