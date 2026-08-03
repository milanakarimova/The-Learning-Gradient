import numpy as np


def train_test_split_numpy(X_num, categories, y, test_size=0.25, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    test_n = int(n * test_size)

    test_idx = idx[:test_n]
    train_idx = idx[test_n:]

    return (
        X_num[train_idx],
        X_num[test_idx],
        categories[train_idx],
        categories[test_idx],
        y[train_idx],
        y[test_idx],
    )


def fit_standardizer(X_train):
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0)
    std = np.where(std == 0, 1, std)
    return mean, std


def transform_standardizer(X, mean, std):
    return (X - mean) / std


def fit_median_imputer(X_train):
    return np.nanmedian(X_train, axis=0)


def transform_median_imputer(X, medians):
    missing_flags = np.isnan(X).astype(float)
    X_filled = np.where(np.isnan(X), medians, X)
    return X_filled, missing_flags


def fit_one_hot(categories):
    unique = sorted(set(categories))
    return {cat: i for i, cat in enumerate(unique)}


def transform_one_hot(categories, mapping):
    X = np.zeros((len(categories), len(mapping)))

    for row, cat in enumerate(categories):
        if cat in mapping:
            X[row, mapping[cat]] = 1

    return X


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def train_logistic_regression(X, y, lr=0.1, steps=1200):
    X_bias = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.zeros(X_bias.shape[1])

    for _ in range(steps):
        p = sigmoid(X_bias @ beta)
        gradient = (1 / len(y)) * X_bias.T @ (p - y)
        beta -= lr * gradient

    return beta


def predict(X, beta, threshold=0.5):
    X_bias = np.column_stack([np.ones(X.shape[0]), X])
    p = sigmoid(X_bias @ beta)
    return (p >= threshold).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# ---------------------------------------------------------
# 1. Create a small mixed-type dataset
# ---------------------------------------------------------
rng = np.random.default_rng(42)

n = 260

age = rng.normal(35, 9, size=n)
income = rng.lognormal(mean=8.2, sigma=0.45, size=n)
transactions_30d = rng.poisson(8, size=n).astype(float)

# Inject missing values
income[rng.choice(n, size=30, replace=False)] = np.nan
transactions_30d[rng.choice(n, size=20, replace=False)] = np.nan

device = rng.choice(["ios", "android", "web"], size=n, p=[0.35, 0.45, 0.20])

# A true signal that uses numeric and categorical information.
income_signal = np.nan_to_num(np.log1p(income), nan=np.nanmedian(np.log1p(income)))
tx_signal = np.nan_to_num(transactions_30d, nan=np.nanmedian(transactions_30d))

logit = (
    -5.5
    + 0.05 * age
    + 0.35 * income_signal
    - 0.10 * tx_signal
    + 0.55 * (device == "web")
)

prob = sigmoid(logit)
y = rng.binomial(1, prob)

X_num = np.column_stack([age, income, transactions_30d])

print("Dataset definition")
print("------------------")
print("row unit: one user at prediction time")
print("target: whether the user has the positive event in the future window")
print("allowed features: age, income known at cutoff, transactions before cutoff, device")
print("forbidden features: anything measured after prediction time")

# ---------------------------------------------------------
# 2. Split first
# ---------------------------------------------------------
X_train_num, X_test_num, cat_train, cat_test, y_train, y_test = train_test_split_numpy(
    X_num,
    device,
    y,
    test_size=0.25,
    seed=42,
)

# ---------------------------------------------------------
# 3. Correct preprocessing: fit only on train
# ---------------------------------------------------------
medians = fit_median_imputer(X_train_num)
X_train_filled, train_missing_flags = transform_median_imputer(X_train_num, medians)
X_test_filled, test_missing_flags = transform_median_imputer(X_test_num, medians)

mean, std = fit_standardizer(X_train_filled)
X_train_scaled = transform_standardizer(X_train_filled, mean, std)
X_test_scaled = transform_standardizer(X_test_filled, mean, std)

mapping = fit_one_hot(cat_train)
X_train_cat = transform_one_hot(cat_train, mapping)
X_test_cat = transform_one_hot(cat_test, mapping)

# Interaction feature
train_interaction = (X_train_scaled[:, [0]] * X_train_scaled[:, [2]])
test_interaction = (X_test_scaled[:, [0]] * X_test_scaled[:, [2]])

X_train_final = np.column_stack([
    X_train_scaled,
    train_missing_flags,
    X_train_cat,
    train_interaction,
])

X_test_final = np.column_stack([
    X_test_scaled,
    test_missing_flags,
    X_test_cat,
    test_interaction,
])

beta = train_logistic_regression(X_train_final, y_train)
pred = predict(X_test_final, beta)

print()
print("Correct train-only preprocessing")
print("--------------------------------")
print("one-hot mapping:", mapping)
print("train feature shape:", X_train_final.shape)
print("test feature shape :", X_test_final.shape)
print("test accuracy:", round(accuracy(y_test, pred), 4))

# ---------------------------------------------------------
# 4. Leakage demonstration: fitting scaler on full data
# This may not always change accuracy dramatically in a toy dataset,
# but the protocol is wrong because test information affects preprocessing.
# ---------------------------------------------------------
X_all_filled, _ = transform_median_imputer(X_num, np.nanmedian(X_num, axis=0))
mean_full, std_full = fit_standardizer(X_all_filled)

print()
print("Leakage demonstration")
print("---------------------")
print("Correct train mean:", np.round(mean, 3))
print("Leaky full-data mean:", np.round(mean_full, 3))
print("These are different because full-data preprocessing used test distribution.")

# ---------------------------------------------------------
# 5. Target leakage example
# ---------------------------------------------------------
future_label_copy = y.copy()

print()
print("Target leakage example")
print("----------------------")
print("A forbidden feature like future_label_copy would directly contain the answer.")
print("It may produce excellent validation scores, but it would not exist at prediction time.")

# ---------------------------------------------------------
# 6. Temporal leakage example
# ---------------------------------------------------------
print()
print("Temporal leakage example")
print("------------------------")
print("If prediction time is Jan 1, then a feature like transactions_after_Jan_1 is forbidden.")
print("Only information available up to Jan 1 can be used.")
