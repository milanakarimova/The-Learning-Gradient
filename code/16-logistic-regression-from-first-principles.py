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

    return X_train_scaled, X_test_scaled, mean, std


def add_bias_column(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true, p_pred, eps=1e-15):
    p_pred = np.clip(p_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(p_pred)
        + (1 - y_true) * np.log(1 - p_pred)
    )


def train_logistic_regression_gd(X, y, lr=0.1, steps=3000, lambda_=0.0):
    X_bias = add_bias_column(X)
    beta = np.zeros(X_bias.shape[1])
    losses = []

    for step in range(steps):
        logits = X_bias @ beta
        probabilities = sigmoid(logits)

        loss = binary_cross_entropy(y, probabilities)
        loss += lambda_ * np.sum(beta[1:] ** 2)
        losses.append(loss)

        gradient = (1 / len(y)) * X_bias.T @ (probabilities - y)

        regularization = np.zeros_like(beta)
        regularization[1:] = 2 * lambda_ * beta[1:]

        beta = beta - lr * (gradient + regularization)

    return beta, np.array(losses)


def predict_proba(X, beta):
    X_bias = add_bias_column(X)
    return sigmoid(X_bias @ beta)


def predict_class(X, beta, threshold=0.5):
    probabilities = predict_proba(X, beta)
    return (probabilities >= threshold).astype(int)


def confusion_counts(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp, tn, fp, fn


def classification_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)

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


def print_metrics(name, metrics):
    print(name)
    print("TP:", metrics["TP"], "TN:", metrics["TN"], "FP:", metrics["FP"], "FN:", metrics["FN"])
    print("Accuracy :", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall   :", round(metrics["recall"], 4))
    print("F1       :", round(metrics["f1"], 4))


rng = np.random.default_rng(42)

# ---------------------------------------------------------
# 1. Create synthetic binary classification dataset
# ---------------------------------------------------------
n = 260

class0 = rng.multivariate_normal(
    mean=[-1.5, -1.0],
    cov=[[0.9, 0.25], [0.25, 0.8]],
    size=n // 2,
)

class1 = rng.multivariate_normal(
    mean=[1.4, 1.2],
    cov=[[0.9, -0.2], [-0.2, 0.9]],
    size=n // 2,
)

X = np.vstack([class0, class1])
y = np.array([0] * (n // 2) + [1] * (n // 2))

# ---------------------------------------------------------
# 2. Split and scale
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.25)
X_train_scaled, X_test_scaled, mean, std = standardize_train_test(X_train, X_test)

# ---------------------------------------------------------
# 3. Train Logistic Regression
# ---------------------------------------------------------
beta, losses = train_logistic_regression_gd(
    X_train_scaled,
    y_train,
    lr=0.3,
    steps=3000,
    lambda_=0.01,
)

# ---------------------------------------------------------
# 4. Predict probabilities
# ---------------------------------------------------------
probabilities = predict_proba(X_test_scaled, beta)
test_loss = binary_cross_entropy(y_test, probabilities)

print("Learned beta:")
print(np.round(beta, 4))

print()
print("Training loss")
print("first:", round(losses[0], 4))
print("last :", round(losses[-1], 4))

print()
print("Test binary cross-entropy:")
print(round(test_loss, 4))

# ---------------------------------------------------------
# 5. Threshold comparison
# ---------------------------------------------------------
for threshold in [0.3, 0.5, 0.7]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = classification_metrics(y_test, predictions)

    print()
    print_metrics(f"Metrics at threshold {threshold}", metrics)

# ---------------------------------------------------------
# 6. Baseline model
# ---------------------------------------------------------
majority_class = int(np.round(np.mean(y_train)))
baseline_predictions = np.ones_like(y_test) * majority_class
baseline_metrics = classification_metrics(y_test, baseline_predictions)

print()
print_metrics("Majority-class baseline", baseline_metrics)

# ---------------------------------------------------------
# 7. Probability examples
# ---------------------------------------------------------
print()
print("First 10 predicted probabilities:")
print(np.round(probabilities[:10], 4))

print()
print("First 10 true labels:")
print(y_test[:10])
