import numpy as np


def predict(X, w, b):
    return X @ w + b


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gradients(X, y, w, b):
    n = X.shape[0]
    y_pred = predict(X, w, b)
    errors = y - y_pred

    dL_dw = -(2 / n) * (X.T @ errors)
    dL_db = -(2 / n) * np.sum(errors)

    return dL_dw, dL_db


def train_gradient_descent(X, y, learning_rate=0.01, steps=1000):
    n_features = X.shape[1]

    w = np.zeros(n_features)
    b = 0.0

    for step in range(steps + 1):
        y_pred = predict(X, w, b)
        loss = mse(y, y_pred)

        if step % 100 == 0:
            print(f"step={step:4d}, loss={loss:.5f}, w={w}, b={b:.5f}")

        dL_dw, dL_db = gradients(X, y, w, b)

        w = w - learning_rate * dL_dw
        b = b - learning_rate * dL_db

    return w, b


x = np.array([1, 2, 3, 4, 5], dtype=float)
y = 2 * x + 1

X = x.reshape(-1, 1)

print("Training simple linear regression with gradient descent")
w, b = train_gradient_descent(X, y, learning_rate=0.01, steps=1000)

print()
print("Final parameters")
print("w:", w)
print("b:", b)

print()
print("Comparing learning rates")

for learning_rate in [0.001, 0.01, 0.1]:
    print()
    print("Learning rate:", learning_rate)
    w_lr, b_lr = train_gradient_descent(X, y, learning_rate=learning_rate, steps=300)
