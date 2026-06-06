import numpy as np


def linear_predict_one(x, w, b):
    return np.dot(w, x) + b


def linear_predict_batch(X, w, b):
    return X @ w + b


def mean_squared_error(y_true, y_pred):
    errors = y_true - y_pred
    return np.mean(errors ** 2)


x = np.array([120, 3, 5], dtype=float)
w = np.array([1000, 15000, -3000], dtype=float)
b = 50000

manual_prediction = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b
dot_prediction = linear_predict_one(x, w, b)

print("Manual prediction:", manual_prediction)
print("Dot product prediction:", dot_prediction)

X = np.array([
    [120, 3, 5],
    [80, 2, 10],
    [200, 5, 2],
    [150, 4, 7],
], dtype=float)

y = np.array([180000, 120000, 300000, 240000], dtype=float)

y_pred = linear_predict_batch(X, w, b)

print()
print("Batch predictions:")
print(y_pred)

errors = y - y_pred
mse = mean_squared_error(y, y_pred)

print()
print("Residuals:")
print(errors)

print()
print("MSE:")
print(mse)

w_changed = np.array([1200, 15000, -3000], dtype=float)
y_pred_changed = linear_predict_batch(X, w_changed, b)

print()
print("Predictions after changing size weight:")
print(y_pred_changed)

print()
print("Prediction difference:")
print(y_pred_changed - y_pred)
