import numpy as np


def train_test_split_numpy(X, y, test_size=0.25, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.permutation(n)

    test_n = int(n * test_size)
    test_idx = indices[:test_n]
    train_idx = indices[test_n:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def add_bias_column(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def predict_linear(X, beta):
    X_bias = add_bias_column(X)
    return X_bias @ beta


def linear_regression_pinv(X, y):
    X_bias = add_bias_column(X)
    beta = np.linalg.pinv(X_bias) @ y
    return beta


def linear_regression_gradient_descent(X, y, lr=0.05, steps=3000):
    X_bias = add_bias_column(X)
    beta = np.zeros(X_bias.shape[1])

    losses = []

    for step in range(steps):
        y_pred = X_bias @ beta
        error = y_pred - y

        loss = np.mean(error ** 2)
        losses.append(loss)

        gradient = (2 / len(y)) * X_bias.T @ error
        beta = beta - lr * gradient

    return beta, np.array(losses)


def ridge_regression_closed_form(X, y, lambda_=1.0):
    X_bias = add_bias_column(X)
    identity = np.eye(X_bias.shape[1])

    # Usually we do not regularize the intercept.
    identity[0, 0] = 0

    beta = np.linalg.solve(
        X_bias.T @ X_bias + lambda_ * identity,
        X_bias.T @ y
    )

    return beta


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - ss_res / ss_tot


def regression_report(y_true, y_pred, name="model"):
    return {
        "model": name,
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def print_report(report):
    print(report["model"])
    print("MAE :", round(report["MAE"], 4))
    print("MSE :", round(report["MSE"], 4))
    print("RMSE:", round(report["RMSE"], 4))
    print("R2  :", round(report["R2"], 4))


rng = np.random.default_rng(42)

# ---------------------------------------------------------
# 1. Create synthetic regression dataset
# ---------------------------------------------------------
n = 180
X = rng.normal(0, 1, size=(n, 3))

true_beta = np.array([4.0, 2.5, -1.2, 0.8])  # intercept, w1, w2, w3

noise = rng.normal(0, 1.0, size=n)
y = true_beta[0] + X @ true_beta[1:] + noise

# ---------------------------------------------------------
# 2. Split into train and test
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.25)

# ---------------------------------------------------------
# 3. Mean baseline
# ---------------------------------------------------------
baseline_prediction = np.ones_like(y_test) * np.mean(y_train)

baseline_report = regression_report(
    y_test,
    baseline_prediction,
    name="Mean baseline"
)

# ---------------------------------------------------------
# 4. Pseudo-inverse Linear Regression
# ---------------------------------------------------------
beta_pinv = linear_regression_pinv(X_train, y_train)
pred_pinv = predict_linear(X_test, beta_pinv)

pinv_report = regression_report(
    y_test,
    pred_pinv,
    name="Linear Regression with pseudo-inverse"
)

# ---------------------------------------------------------
# 5. Gradient descent Linear Regression
# ---------------------------------------------------------
# Standardization helps gradient descent.
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std

beta_gd, losses = linear_regression_gradient_descent(
    X_train_scaled,
    y_train,
    lr=0.05,
    steps=3000
)

pred_gd = predict_linear(X_test_scaled, beta_gd)

gd_report = regression_report(
    y_test,
    pred_gd,
    name="Linear Regression with gradient descent"
)

# ---------------------------------------------------------
# 6. Ridge Regression
# ---------------------------------------------------------
beta_ridge = ridge_regression_closed_form(X_train_scaled, y_train, lambda_=3.0)
pred_ridge = predict_linear(X_test_scaled, beta_ridge)

ridge_report = regression_report(
    y_test,
    pred_ridge,
    name="Ridge Regression"
)

# ---------------------------------------------------------
# 7. Print results
# ---------------------------------------------------------
print("True beta:")
print(true_beta)

print()
print("Pseudo-inverse beta:")
print(np.round(beta_pinv, 4))

print()
print("Gradient descent beta on scaled features:")
print(np.round(beta_gd, 4))

print()
print("Ridge beta on scaled features:")
print(np.round(beta_ridge, 4))

print()
print("Evaluation reports")
print("------------------")

for report in [baseline_report, pinv_report, gd_report, ridge_report]:
    print()
    print_report(report)

print()
print("Gradient descent loss")
print("first loss:", round(losses[0], 4))
print("last loss :", round(losses[-1], 4))

# ---------------------------------------------------------
# 8. Residual diagnostics
# ---------------------------------------------------------
residuals = y_test - pred_pinv

print()
print("Residual summary for pseudo-inverse model")
print("mean residual:", round(np.mean(residuals), 4))
print("std residual :", round(np.std(residuals), 4))
print("largest absolute residual:", round(np.max(np.abs(residuals)), 4))

# ---------------------------------------------------------
# 9. Outlier sensitivity demo
# ---------------------------------------------------------
X_out = X_train.copy()
y_out = y_train.copy()

y_out[0] = y_out[0] + 30  # artificial outlier

beta_outlier = linear_regression_pinv(X_out, y_out)
pred_outlier = predict_linear(X_test, beta_outlier)

outlier_report = regression_report(
    y_test,
    pred_outlier,
    name="Linear Regression after artificial outlier"
)

print()
print("Outlier sensitivity")
print("-------------------")
print_report(outlier_report)
