import numpy as np


X = np.array([
    [50, 2],
    [80, 3],
    [120, 4],
    [150, 5],
], dtype=float)

print("Feature matrix X:")
print(X)

print()
print("Shape of X:", X.shape)

feature_means = np.mean(X, axis=0)
feature_stds = np.std(X, axis=0)

print()
print("Feature means:", feature_means)
print("Feature standard deviations:", feature_stds)

X_scaled = (X - feature_means) / feature_stds

print()
print("Standardized X:")
print(X_scaled)

w = np.array([0.3, 10])
b = 5

y_pred = X @ w + b

print()
print("Predictions:")
print(y_pred)

y_true = np.array([40, 60, 85, 105])

errors = y_true - y_pred
mse = np.mean(errors ** 2)

print()
print("True values:")
print(y_true)

print()
print("Errors:")
print(errors)

print()
print("MSE:", mse)
