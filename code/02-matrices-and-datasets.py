import numpy as np


X = np.array([
    [120, 3, 5],
    [80, 2, 10],
    [200, 5, 2],
    [150, 4, 7],
], dtype=float)

y = np.array([180000, 120000, 300000, 240000], dtype=float)

print("Feature matrix X:")
print(X)
print("Shape of X:", X.shape)

print()
print("Target vector y:")
print(y)
print("Shape of y:", y.shape)

w = np.array([1000, 15000, -3000], dtype=float)
b = 50000

y_pred = X @ w + b

print()
print("Predictions:")
print(y_pred)
print("Shape of predictions:", y_pred.shape)

errors = y - y_pred
mse = np.mean(errors ** 2)

print()
print("Errors:")
print(errors)
print("MSE:", mse)

ones = np.ones((X.shape[0], 1))
X_augmented = np.hstack([ones, X])

print()
print("Augmented design matrix:")
print(X_augmented)
print("Shape of augmented matrix:", X_augmented.shape)

feature_means = X.mean(axis=0)
X_centered = X - feature_means

print()
print("Feature means:")
print(feature_means)

print()
print("Centered X:")
print(X_centered)

covariance_like = X_centered.T @ X_centered

print()
print("Covariance-like matrix X_centered.T @ X_centered:")
print(covariance_like)
print("Shape:", covariance_like.shape)
