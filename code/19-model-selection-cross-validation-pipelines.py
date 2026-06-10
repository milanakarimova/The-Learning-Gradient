import numpy as np


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))


def train_test_split_numpy(X, y, test_size=0.25, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    test_n = int(n * test_size)
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1, std)
    return (X_train - mean) / std, (X_test - mean) / std


def kfold_indices(n_samples, k=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, k)

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


def train_logistic_regression(X, y, lr=0.1, steps=1000, lambda_=0.0):
    X_bias = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.zeros(X_bias.shape[1])

    for _ in range(steps):
        logits = X_bias @ beta
        p = sigmoid(logits)
        gradient = (1 / len(y)) * X_bias.T @ (p - y)

        regularization = np.zeros_like(beta)
        regularization[1:] = 2 * lambda_ * beta[1:]

        beta = beta - lr * (gradient + regularization)

    return beta


def predict_logistic(X, beta, threshold=0.5):
    X_bias = np.column_stack([np.ones(X.shape[0]), X])
    p = sigmoid(X_bias @ beta)
    return (p >= threshold).astype(int)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def f1_score_binary(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def cross_validate_lambda(X, y, lambda_, k=5):
    scores = []

    for train_idx, val_idx in kfold_indices(len(y), k=k, seed=42):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # Fit preprocessing on training fold only.
        X_train_scaled, X_val_scaled = standardize_train_test(X_train_fold, X_val_fold)

        beta = train_logistic_regression(
            X_train_scaled,
            y_train_fold,
            lr=0.2,
            steps=1200,
            lambda_=lambda_,
        )

        pred = predict_logistic(X_val_scaled, beta)
        scores.append(f1_score_binary(y_val_fold, pred))

    return np.array(scores)


# ---------------------------------------------------------
# 1. Synthetic classification dataset
# ---------------------------------------------------------
rng = np.random.default_rng(42)

n = 320
X0 = rng.multivariate_normal([-1.0, -0.8], [[1.0, 0.35], [0.35, 1.0]], size=n // 2)
X1 = rng.multivariate_normal([1.1, 0.9], [[1.0, -0.25], [-0.25, 1.0]], size=n // 2)

X = np.vstack([X0, X1])
y = np.array([0] * (n // 2) + [1] * (n // 2))

X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.25)

# ---------------------------------------------------------
# 2. Baseline model
# ---------------------------------------------------------
majority_class = int(np.round(y_train.mean()))
baseline_pred = np.ones_like(y_test) * majority_class

print("Baseline")
print("--------")
print("accuracy:", round(accuracy_score(y_test, baseline_pred), 4))
print("f1      :", round(f1_score_binary(y_test, baseline_pred), 4))

# ---------------------------------------------------------
# 3. Cross-validation from scratch for lambda selection
# ---------------------------------------------------------
lambda_values = [0.0, 0.001, 0.01, 0.1, 1.0]

print()
print("From-scratch cross-validation")
print("-----------------------------")

cv_results = {}

for lambda_ in lambda_values:
    scores = cross_validate_lambda(X_train, y_train, lambda_=lambda_, k=5)
    cv_results[lambda_] = scores
    print(
        "lambda=", lambda_,
        "mean F1=", round(scores.mean(), 4),
        "std=", round(scores.std(), 4),
    )

best_lambda = max(cv_results, key=lambda value: cv_results[value].mean())

print()
print("best lambda from CV:", best_lambda)

# ---------------------------------------------------------
# 4. Train final from-scratch model on full training set
# ---------------------------------------------------------
X_train_scaled, X_test_scaled = standardize_train_test(X_train, X_test)

final_beta = train_logistic_regression(
    X_train_scaled,
    y_train,
    lr=0.2,
    steps=1500,
    lambda_=best_lambda,
)

final_pred = predict_logistic(X_test_scaled, final_beta)

print()
print("Final test evaluation from scratch")
print("----------------------------------")
print("accuracy:", round(accuracy_score(y_test, final_pred), 4))
print("f1      :", round(f1_score_binary(y_test, final_pred), 4))

# ---------------------------------------------------------
# 5. Optional Scikit-learn Pipeline + GridSearchCV
# ---------------------------------------------------------
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import accuracy_score as sk_accuracy_score
    from sklearn.metrics import f1_score as sk_f1_score

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC())
    ])

    param_grid = {
        "svm__kernel": ["linear", "rbf"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", 0.1, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
    )

    search.fit(X_train, y_train)

    test_pred = search.predict(X_test)

    print()
    print("Scikit-learn Pipeline + GridSearchCV")
    print("------------------------------------")
    print("best params:", search.best_params_)
    print("best CV F1 :", round(search.best_score_, 4))
    print("test acc   :", round(sk_accuracy_score(y_test, test_pred), 4))
    print("test F1    :", round(sk_f1_score(y_test, test_pred), 4))

except Exception as exc:
    print()
    print("Scikit-learn part skipped:")
    print(exc)

# ---------------------------------------------------------
# 6. Leakage reminder
# ---------------------------------------------------------
print()
print("Leakage reminder")
print("----------------")
print("Wrong: fit scaler on full data before CV.")
print("Right: fit scaler inside each training fold or use Pipeline.")
