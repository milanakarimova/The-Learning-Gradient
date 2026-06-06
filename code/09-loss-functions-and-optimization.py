import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def huber_loss(y_true, y_pred, delta=1.0):
    residual = y_true - y_pred
    abs_residual = np.abs(residual)

    quadratic = 0.5 * residual ** 2
    linear = delta * (abs_residual - 0.5 * delta)

    return np.mean(np.where(abs_residual <= delta, quadratic, linear))


def binary_cross_entropy(y_true, p_pred, eps=1e-15):
    p_pred = np.clip(p_pred, eps, 1 - eps)
    loss = -(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))
    return np.mean(loss)


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def softmax_cross_entropy(y_true_indices, logits, eps=1e-15):
    probabilities = softmax(logits)
    n = logits.shape[0]

    correct_probs = probabilities[np.arange(n), y_true_indices]
    correct_probs = np.clip(correct_probs, eps, 1 - eps)

    return np.mean(-np.log(correct_probs))


def hinge_loss(y_true, scores):
    margins = y_true * scores
    return np.mean(np.maximum(0, 1 - margins))


def l1_penalty(w):
    return np.sum(np.abs(w))


def l2_penalty(w):
    return np.sum(w ** 2)


def train_linear_regression_gd(x, y, learning_rate=0.01, steps=1000):
    w = 0.0
    b = 0.0

    for step in range(steps + 1):
        y_pred = w * x + b
        loss = mse(y, y_pred)

        if step % 200 == 0:
            print(f"step={step:4d}, loss={loss:.5f}, w={w:.4f}, b={b:.4f}")

        errors = y - y_pred

        dL_dw = -(2 / len(x)) * np.sum(x * errors)
        dL_db = -(2 / len(x)) * np.sum(errors)

        w = w - learning_rate * dL_dw
        b = b - learning_rate * dL_db

    return w, b


y_true = np.array([3, 5, 7, 9], dtype=float)
y_pred_good = np.array([2.8, 5.2, 6.9, 9.1])
y_pred_outlier = np.array([2.8, 5.2, 6.9, 20.0])

print("Regression losses with good predictions")
print("MSE:", mse(y_true, y_pred_good))
print("MAE:", mae(y_true, y_pred_good))
print("Huber:", huber_loss(y_true, y_pred_good))

print()
print("Regression losses with one outlier prediction")
print("MSE:", mse(y_true, y_pred_outlier))
print("MAE:", mae(y_true, y_pred_outlier))
print("Huber:", huber_loss(y_true, y_pred_outlier))

y_binary = np.array([1, 0, 1, 1, 0])
p_binary = np.array([0.9, 0.1, 0.8, 0.4, 0.2])

print()
print("Binary cross-entropy:")
print(binary_cross_entropy(y_binary, p_binary))

logits = np.array([
    [2.0, 0.5, -1.0],
    [0.2, 1.8, 0.1],
    [-1.0, 0.5, 2.5],
])

y_classes = np.array([0, 1, 2])

print()
print("Softmax probabilities:")
print(softmax(logits))

print()
print("Softmax cross-entropy:")
print(softmax_cross_entropy(y_classes, logits))

y_hinge = np.array([1, -1, 1, -1])
scores = np.array([2.0, -0.3, 0.2, 1.1])

print()
print("Hinge loss:")
print(hinge_loss(y_hinge, scores))

w = np.array([2.0, -1.0, 0.5])

print()
print("Regularization penalties")
print("L1:", l1_penalty(w))
print("L2:", l2_penalty(w))

print()
print("Training linear regression with gradient descent")
x_train = np.array([1, 2, 3, 4, 5], dtype=float)
y_train = 2 * x_train + 1

final_w, final_b = train_linear_regression_gd(x_train, y_train)

print()
print("Final parameters")
print("w:", final_w)
print("b:", final_b)
