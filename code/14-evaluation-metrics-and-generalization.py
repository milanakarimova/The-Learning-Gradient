import numpy as np


def confusion_counts(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp, fp, tn, fn


def classification_metrics(y_true, y_pred):
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)

    accuracy = (tp + tn) / (tp + fp + tn + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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


def bootstrap_accuracy_ci(y_true, y_pred, n_boot=3000, confidence=0.95, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    scores = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(np.mean(y_true[idx] == y_pred[idx]))

    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return lower, upper


def threshold_sweep(y_true, probabilities, thresholds):
    results = []

    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        metrics = classification_metrics(y_true, y_pred)
        results.append({
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

    return results


def detect_overfitting(train_losses, val_losses, patience=5):
    best_val_index = int(np.argmin(val_losses))

    train_end = train_losses[-1]
    val_end = val_losses[-1]

    if val_end > val_losses[best_val_index] and len(val_losses) - best_val_index > patience:
        return "Possible overfitting: validation loss worsened after its best point."

    if train_end > 0.9 * train_losses[0] and val_end > 0.9 * val_losses[0]:
        return "Possible underfitting: both train and validation loss remain high."

    return "No obvious overfitting pattern in this simple check."


# Imbalanced classification example
y_true = np.array([0] * 90 + [1] * 10)

# Model predicts all zeros
y_pred_all_negative = np.array([0] * 100)

# Better model catches some positives but has some false positives
y_pred_better = np.array([0] * 85 + [1] * 5 + [1] * 7 + [0] * 3)

print("Imbalanced classification example")
print()
print("All-negative model:")
print(classification_metrics(y_true, y_pred_all_negative))

print()
print("Better positive-detecting model:")
print(classification_metrics(y_true, y_pred_better))

ci = bootstrap_accuracy_ci(y_true, y_pred_better)

print()
print("Bootstrap accuracy CI for better model:")
print(ci)

probabilities = np.concatenate([
    np.linspace(0.01, 0.45, 90),
    np.linspace(0.35, 0.95, 10),
])

thresholds = np.linspace(0.2, 0.8, 7)
sweep = threshold_sweep(y_true, probabilities, thresholds)

print()
print("Threshold sweep:")
for row in sweep:
    print(row)

# Regression example
y_reg_true = np.array([10, 12, 15, 18, 20, 24], dtype=float)
y_reg_pred = np.array([9, 13, 14, 17, 23, 22], dtype=float)

print()
print("Regression metrics")
print("MAE:", mae(y_reg_true, y_reg_pred))
print("MSE:", mse(y_reg_true, y_reg_pred))
print("RMSE:", rmse(y_reg_true, y_reg_pred))
print("R2:", r2_score(y_reg_true, y_reg_pred))

# Simulated loss curves
epochs = np.arange(1, 61)
train_losses = 2.0 * np.exp(-epochs / 25) + 0.1
val_losses = 1.5 * np.exp(-epochs / 22) + 0.25 + 0.01 * np.maximum(epochs - 30, 0)

print()
print("Loss curve diagnosis:")
print(detect_overfitting(train_losses, val_losses))
