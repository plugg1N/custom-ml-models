import random

# Data
X = [i for i in range(10)]
y = [i*2 for i in X]

# Constants
b0 = 0                      # bias
alpha = 1e-1                # learning rate
eps = 1e-1                  # epsilon
w = random.randint(0, 20)   # random weight for init


# Calculate the function
def h(x: int, w: float, b0: int) -> int:
    return (x * w) + b0


# Calculate cost function
def cost(y_true: list, y_preds: list) -> float:
    sums = 0
    for i in range(len(y_true)):
        sums += (y_true[i] - y_preds[i]) ** 2
    return sums / len(y_true)


"""
# Gradient descent
def gradient_descent(X: list, y: list, w: int):
    preds = []
    for i in range(len(X)):
        preds.append( h(X[i], w, b0) )
    J = cost(y, preds)

    
    while J > 0:
        preds = []

        abomination = alpha * ( (h(X[i]+eps, w, b0) + h(X[i], w, b0)) / eps )
        tmp = w - abomination
        w = tmp

        for j in range(len(X)):
            preds.append( h(X[j], w, b0) )

        J = cost(y, preds)
        print(J)
        
    return w
"""


