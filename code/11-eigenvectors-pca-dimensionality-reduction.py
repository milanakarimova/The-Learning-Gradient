import numpy as np


def pca_fit(X, n_components):
    mean = X.mean(axis=0)
    X_centered = X - mean

    covariance = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]
    explained_variance_ratio = explained_variance / eigenvalues.sum()

    return {
        "mean": mean,
        "components": components,
        "eigenvalues": eigenvalues,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "covariance": covariance,
    }


def pca_transform(X, mean, components):
    X_centered = X - mean
    return X_centered @ components


def pca_inverse_transform(Z, mean, components):
    return Z @ components.T + mean


def reconstruction_error(X, X_reconstructed):
    return np.mean(np.sum((X - X_reconstructed) ** 2, axis=1))


rng = np.random.default_rng(42)

mean = np.array([0, 0])
covariance_true = np.array([
    [3.0, 1.8],
    [1.8, 1.4],
])

X = rng.multivariate_normal(mean, covariance_true, size=300)

result = pca_fit(X, n_components=1)

print("Feature means:")
print(result["mean"])

print()
print("Covariance matrix:")
print(result["covariance"])

print()
print("Eigenvalues:")
print(result["eigenvalues"])

print()
print("First principal component:")
print(result["components"])

print()
print("Explained variance ratio:")
print(result["explained_variance_ratio"])

Z = pca_transform(X, result["mean"], result["components"])
X_reconstructed = pca_inverse_transform(Z, result["mean"], result["components"])

error = reconstruction_error(X, X_reconstructed)

print()
print("Shape of original X:", X.shape)
print("Shape of reduced Z:", Z.shape)
print("Reconstruction error with 1 component:", error)

# Compare PCA before and after standardization
X_scaled_problem = X.copy()
X_scaled_problem[:, 1] = X_scaled_problem[:, 1] * 20

pca_unscaled = pca_fit(X_scaled_problem, n_components=2)

X_standardized = (X_scaled_problem - X_scaled_problem.mean(axis=0)) / X_scaled_problem.std(axis=0)
pca_standardized = pca_fit(X_standardized, n_components=2)

print()
print("PCA components before standardization:")
print(pca_unscaled["components"])

print()
print("Explained variance ratio before standardization:")
print(pca_unscaled["explained_variance_ratio"])

print()
print("PCA components after standardization:")
print(pca_standardized["components"])

print()
print("Explained variance ratio after standardization:")
print(pca_standardized["explained_variance_ratio"])
