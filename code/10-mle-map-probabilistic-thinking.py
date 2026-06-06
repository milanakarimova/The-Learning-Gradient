import numpy as np


def bernoulli_log_likelihood(p, data, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    data = np.array(data)
    return np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))


def bernoulli_mle(data):
    data = np.array(data)
    return np.mean(data)


def bernoulli_map_beta(data, alpha, beta):
    data = np.array(data)
    n = len(data)
    k = np.sum(data)

    if alpha + k <= 1 or beta + n - k <= 1:
        raise ValueError("MAP formula requires posterior alpha and beta greater than 1.")

    return (alpha + k - 1) / (alpha + beta + n - 2)


def gaussian_nll(y_true, y_pred, sigma=1.0):
    residual = y_true - y_pred
    return np.mean(
        0.5 * np.log(2 * np.pi * sigma ** 2)
        + (residual ** 2) / (2 * sigma ** 2)
    )


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def l2_penalty(w):
    return np.sum(w ** 2)


def map_objective_mse_l2(y_true, y_pred, w, lambda_):
    return mse(y_true, y_pred) + lambda_ * l2_penalty(w)


data_small = np.array([1, 1, 1, 0, 0])
data_large = np.array([1] * 70 + [0] * 30)

print("Bernoulli MLE")
print("small data:", bernoulli_mle(data_small))
print("large data:", bernoulli_mle(data_large))

alpha = 2
beta = 5

print()
print("Bernoulli MAP with Beta prior")
print("small data:", bernoulli_map_beta(data_small, alpha, beta))
print("large data:", bernoulli_map_beta(data_large, alpha, beta))

candidate_ps = np.linspace(0.05, 0.95, 19)
log_likelihoods = np.array([
    bernoulli_log_likelihood(p, data_small)
    for p in candidate_ps
])

best_index = np.argmax(log_likelihoods)

print()
print("Candidate p values:")
print(candidate_ps)

print()
print("Log-likelihoods for small data:")
print(np.round(log_likelihoods, 3))

print()
print("Best grid p:")
print(candidate_ps[best_index])

y_true = np.array([3, 5, 7, 9], dtype=float)
y_pred = np.array([2.8, 5.2, 6.9, 9.1], dtype=float)

print()
print("Gaussian NLL and MSE")
print("Gaussian NLL:", gaussian_nll(y_true, y_pred, sigma=1.0))
print("MSE:", mse(y_true, y_pred))

w = np.array([2.0, -1.0, 0.5])

print()
print("MAP-style objective with L2 penalty")
print(map_objective_mse_l2(y_true, y_pred, w, lambda_=0.1))
