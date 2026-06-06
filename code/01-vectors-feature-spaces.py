import numpy as np


x = np.array([120, 3, 5])

print("Feature vector x:")
print(x)
print("Shape of x:", x.shape)

X = np.array([
    [120, 3, 5],
    [80, 2, 10],
    [200, 5, 2],
    [150, 4, 7],
], dtype=float)

print()
print("Dataset matrix X:")
print(X)
print("Shape of X:", X.shape)

w = np.array([1000, 15000, -3000])
b = 50000

prediction = np.dot(w, x) + b

print()
print("Linear prediction w^T x + b:")
print(prediction)

norm_x = np.linalg.norm(x)

print()
print("Euclidean norm of x:")
print(norm_x)

sample_a = X[0]
sample_b = X[1]

distance = np.linalg.norm(sample_a - sample_b)

print()
print("Distance between first two samples:")
print(distance)

feature_means = X.mean(axis=0)
feature_stds = X.std(axis=0)

X_scaled = (X - feature_means) / feature_stds

print()
print("Feature means:")
print(feature_means)

print()
print("Feature standard deviations:")
print(feature_stds)

print()
print("Standardized dataset:")
print(X_scaled)
