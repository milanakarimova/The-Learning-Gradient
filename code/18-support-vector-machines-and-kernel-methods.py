import numpy as np


def train_test_split_numpy(X, y, test_size=0.25, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.permutation(n)

    test_n = int(n * test_size)
    test_idx = indices[:test_n]
    train_idx = indices[test_n:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled


def hinge_loss(y, scores):
    margins = y * scores
    return np.mean(np.maximum(0, 1 - margins))


def train_linear_svm(X, y, C=1.0, lr=0.01, epochs=2500):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []

    for epoch in range(epochs):
        scores = X @ w + b
        margins = y * scores
        hinge = np.maximum(0, 1 - margins)

        objective = 0.5 * np.dot(w, w) + C * np.mean(hinge)
        losses.append(objective)

        violating = margins < 1

        if np.any(violating):
            grad_w = w - C * np.mean(y[violating, None] * X[violating], axis=0)
            grad_b = -C * np.mean(y[violating])
        else:
            grad_w = w
            grad_b = 0.0

        step = lr / (1 + 0.0005 * epoch)
        w = w - step * grad_w
        b = b - step * grad_b

    return w, b, np.array(losses)


def predict_linear_svm(X, w, b):
    scores = X @ w + b
    return np.where(scores >= 0, 1, -1)


def classification_metrics(y_true, y_pred):
    positive = 1
    negative = -1

    tp = np.sum((y_true == positive) & (y_pred == positive))
    tn = np.sum((y_true == negative) & (y_pred == negative))
    fp = np.sum((y_true == negative) & (y_pred == positive))
    fn = np.sum((y_true == positive) & (y_pred == negative))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def linear_kernel(X, Z):
    return X @ Z.T


def polynomial_kernel(X, Z, degree=3, coef0=1.0):
    return (X @ Z.T + coef0) ** degree


def rbf_kernel(X, Z, gamma=1.0):
    X_norm = np.sum(X ** 2, axis=1)[:, None]
    Z_norm = np.sum(Z ** 2, axis=1)[None, :]
    sq_dist = X_norm + Z_norm - 2 * X @ Z.T
    return np.exp(-gamma * sq_dist)


rng = np.random.default_rng(42)

# ---------------------------------------------------------
# 1. Synthetic binary classification dataset
# ---------------------------------------------------------
n = 220

class_neg = rng.multivariate_normal(
    mean=[-1.8, -0.8],
    cov=[[0.35, 0.08], [0.08, 0.35]],
    size=n // 2,
)

class_pos = rng.multivariate_normal(
    mean=[1.7, 0.9],
    cov=[[0.35, -0.05], [-0.05, 0.35]],
    size=n // 2,
)

X = np.vstack([class_neg, class_pos])
y = np.array([-1] * (n // 2) + [1] * (n // 2))

X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.25)
X_train_scaled, X_test_scaled = standardize_train_test(X_train, X_test)

# ---------------------------------------------------------
# 2. Train linear SVM
# ---------------------------------------------------------
w, b, losses = train_linear_svm(
    X_train_scaled,
    y_train,
    C=2.0,
    lr=0.04,
    epochs=3000,
)

pred = predict_linear_svm(X_test_scaled, w, b)
metrics = classification_metrics(y_test, pred)

print("Linear SVM from scratch")
print("-----------------------")
print("w:", np.round(w, 4))
print("b:", round(b, 4))
print("first objective:", round(losses[0], 4))
print("last objective :", round(losses[-1], 4))

print()
print("Metrics:")
for key, value in metrics.items():
    print(key, round(value, 4) if isinstance(value, float) else value)

# ---------------------------------------------------------
# 3. Approximate support vectors
# ---------------------------------------------------------
train_scores = X_train_scaled @ w + b
train_margins = y_train * train_scores
support_mask = train_margins <= 1.05

print()
print("Approximate support vectors:")
print(np.sum(support_mask), "out of", len(y_train))

print()
print("First 10 margins:")
print(np.round(train_margins[:10], 4))

# ---------------------------------------------------------
# 4. Kernel matrices
# ---------------------------------------------------------
K_linear = linear_kernel(X_train_scaled[:5], X_train_scaled[:5])
K_poly = polynomial_kernel(X_train_scaled[:5], X_train_scaled[:5], degree=3)
K_rbf = rbf_kernel(X_train_scaled[:5], X_train_scaled[:5], gamma=0.7)

print()
print("Kernel matrix examples using first 5 training samples")
print("Linear kernel:")
print(np.round(K_linear, 3))

print()
print("Polynomial kernel:")
print(np.round(K_poly, 3))

print()
print("RBF kernel:")
print(np.round(K_rbf, 3))

# ---------------------------------------------------------
# 5. Optional sklearn comparison
# ---------------------------------------------------------
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=1.0))
    ])

    rbf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])

    # sklearn uses labels as provided, -1 and +1 are fine
    linear_model.fit(X_train, y_train)
    rbf_model.fit(X_train, y_train)

    pred_linear = linear_model.predict(X_test)
    pred_rbf = rbf_model.predict(X_test)

    print()
    print("Sklearn comparison")
    print("------------------")
    print("Linear SVC accuracy:", round(accuracy_score(y_test, pred_linear), 4))
    print("RBF SVC accuracy   :", round(accuracy_score(y_test, pred_rbf), 4))

except Exception as exc:
    print()
    print("Sklearn comparison skipped:")
    print(exc)
