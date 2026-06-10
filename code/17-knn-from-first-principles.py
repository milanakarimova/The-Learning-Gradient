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


def euclidean_distances(X_train, X_query):
    diff = X_query[:, None, :] - X_train[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def knn_predict_classification(X_train, y_train, X_query, k=5):
    distances = euclidean_distances(X_train, X_query)
    neighbor_indices = np.argsort(distances, axis=1)[:, :k]
    neighbor_labels = y_train[neighbor_indices]

    predictions = []

    for labels in neighbor_labels:
        counts = np.bincount(labels.astype(int))
        predictions.append(np.argmax(counts))

    return np.array(predictions)


def knn_predict_regression(X_train, y_train, X_query, k=5):
    distances = euclidean_distances(X_train, X_query)
    neighbor_indices = np.argsort(distances, axis=1)[:, :k]
    neighbor_values = y_train[neighbor_indices]

    return neighbor_values.mean(axis=1)


def knn_predict_weighted_regression(X_train, y_train, X_query, k=5, eps=1e-8):
    distances = euclidean_distances(X_train, X_query)
    neighbor_indices = np.argsort(distances, axis=1)[:, :k]

    predictions = []

    for row_id, idx in enumerate(neighbor_indices):
        neighbor_distances = distances[row_id, idx]
        neighbor_values = y_train[idx]

        weights = 1 / (neighbor_distances + eps)
        prediction = np.sum(weights * neighbor_values) / np.sum(weights)
        predictions.append(prediction)

    return np.array(predictions)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


rng = np.random.default_rng(42)

# ---------------------------------------------------------
# 1. Classification dataset
# ---------------------------------------------------------
n = 240

class0 = rng.multivariate_normal(
    mean=[-1.5, -1.0],
    cov=[[0.8, 0.2], [0.2, 0.7]],
    size=n // 2,
)

class1 = rng.multivariate_normal(
    mean=[1.4, 1.1],
    cov=[[0.9, -0.25], [-0.25, 0.9]],
    size=n // 2,
)

X = np.vstack([class0, class1])
y = np.array([0] * (n // 2) + [1] * (n // 2))

X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.25)

X_train_scaled, X_test_scaled = standardize_train_test(X_train, X_test)

print("KNN Classification")
print("------------------")

for k in [1, 3, 5, 11, 21]:
    pred = knn_predict_classification(X_train_scaled, y_train, X_test_scaled, k=k)
    acc = accuracy_score(y_test, pred)
    print(f"k={k:2d}, accuracy={acc:.4f}")

# ---------------------------------------------------------
# 2. Scaling comparison
# ---------------------------------------------------------
X_bad_scale = X.copy()
X_bad_scale[:, 1] = X_bad_scale[:, 1] * 30

X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split_numpy(
    X_bad_scale,
    y,
    test_size=0.25,
)

pred_unscaled = knn_predict_classification(X_train_bad, y_train_bad, X_test_bad, k=7)
acc_unscaled = accuracy_score(y_test_bad, pred_unscaled)

X_train_good, X_test_good = standardize_train_test(X_train_bad, X_test_bad)
pred_scaled = knn_predict_classification(X_train_good, y_train_bad, X_test_good, k=7)
acc_scaled = accuracy_score(y_test_bad, pred_scaled)

print()
print("Scaling comparison")
print("------------------")
print("accuracy without scaling:", round(acc_unscaled, 4))
print("accuracy with scaling   :", round(acc_scaled, 4))

# ---------------------------------------------------------
# 3. Regression dataset
# ---------------------------------------------------------
x = np.linspace(0, 10, 120)
y_reg = np.sin(x) + 0.18 * x + rng.normal(0, 0.25, size=len(x))

X_reg = x.reshape(-1, 1)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split_numpy(X_reg, y_reg, test_size=0.25)

print()
print("KNN Regression")
print("--------------")

for k in [1, 3, 9, 21]:
    pred_r = knn_predict_regression(X_train_r, y_train_r, X_test_r, k=k)
    error = mae(y_test_r, pred_r)
    print(f"k={k:2d}, MAE={error:.4f}")

pred_weighted = knn_predict_weighted_regression(X_train_r, y_train_r, X_test_r, k=9)
print("weighted k=9, MAE:", round(mae(y_test_r, pred_weighted), 4))

# ---------------------------------------------------------
# 4. Example prediction details
# ---------------------------------------------------------
query = X_test_scaled[:1]
distances = euclidean_distances(X_train_scaled, query)[0]
neighbor_indices = np.argsort(distances)[:5]

print()
print("Example query")
print("-------------")
print("query:", np.round(query[0], 4))
print("nearest neighbor labels:", y_train[neighbor_indices])
print("nearest neighbor distances:", np.round(distances[neighbor_indices], 4))
