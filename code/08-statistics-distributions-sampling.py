import numpy as np


def descriptive_statistics(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1

    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "sample_variance": np.var(x, ddof=1),
        "sample_std": np.std(x, ddof=1),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
    }


def iqr_outlier_mask(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return (x < lower) | (x > upper)


def bootstrap_mean_ci(x, n_boot=5000, confidence=0.95, seed=42):
    rng = np.random.default_rng(seed)
    boot_means = []

    for _ in range(n_boot):
        boot_sample = rng.choice(x, size=len(x), replace=True)
        boot_means.append(np.mean(boot_sample))

    alpha = 1 - confidence
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return lower, upper


def covariance(x, y):
    return np.cov(x, y, ddof=1)[0, 1]


def correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


def compare_feature_means(train, test):
    return test.mean(axis=0) - train.mean(axis=0)


x = np.array([10, 12, 15, 18, 20, 100], dtype=float)

print("Descriptive statistics:")
print(descriptive_statistics(x))

print()
print("Outlier mask:")
print(iqr_outlier_mask(x))

rng = np.random.default_rng(42)
population = rng.gamma(shape=2.0, scale=2.0, size=10000)

sample_means = []

for _ in range(2000):
    sample = rng.choice(population, size=30, replace=True)
    sample_means.append(np.mean(sample))

sample_means = np.array(sample_means)

print()
print("Population mean:", np.mean(population))
print("Mean of sample means:", np.mean(sample_means))
print("Standard deviation of sample means:", np.std(sample_means))

sample = rng.choice(population, size=80, replace=False)
standard_error = np.std(sample, ddof=1) / np.sqrt(len(sample))
ci_lower, ci_upper = bootstrap_mean_ci(sample)

print()
print("Sample mean:", np.mean(sample))
print("Standard error:", standard_error)
print("Bootstrap 95% CI:", (ci_lower, ci_upper))

a = rng.normal(0, 1, size=200)
b = 0.8 * a + rng.normal(0, 0.4, size=200)

print()
print("Covariance:", covariance(a, b))
print("Correlation:", correlation(a, b))

train = rng.normal(0, 1, size=(500, 3))
test = rng.normal([0.2, 0.0, 1.0], [1.0, 1.0, 1.2], size=(500, 3))

print()
print("Train-test feature mean differences:")
print(compare_feature_means(train, test))
