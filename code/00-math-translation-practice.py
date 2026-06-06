import numpy as np


x = np.array([2, 3, 4])
w = np.array([0.5, 1.0, -0.2])

dot_product = np.dot(w, x)

print("Dot product:", dot_product)

values = np.array([10, 20, 30, 40, 50])

mean = np.mean(values)
variance = np.var(values)
standard_deviation = np.std(values)

print("Mean:", mean)
print("Variance:", variance)
print("Standard deviation:", standard_deviation)

standardized = (values - mean) / standard_deviation

print("Standardized values:", standardized)

y_true = np.array([3, 5, 2, 7])
y_pred = np.array([2.5, 5.5, 2, 8])

errors = y_true - y_pred
mse = np.mean(errors ** 2)

print("Errors:", errors)
print("MSE:", mse)
