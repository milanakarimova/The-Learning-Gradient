import numpy as np


def l1_norm(x):
    return np.sum(np.abs(x))


def l2_norm(x):
    return np.sqrt(np.sum(x ** 2))


def linf_norm(x):
    return np.max(np.abs(x))


def euclidean_distance(a, b):
    return l2_norm(a - b)


def manhattan_distance(a, b):
    return l1_norm(a - b)


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return np.dot(a, b) / (norm_a * norm_b)


def pairwise_euclidean_distances(X):
    n = X.shape[0]
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            D[i, j] = euclidean_distance(X[i], X[j])

    return D


def nearest_neighbor(X_train, query):
    distances = np.linalg.norm(X_train - query, axis=1)
    index = np.argmin(distances)
    return index, distances[index]


x = np.array([3, -4, 12], dtype=float)

print("Vector x:", x)
print("L1 norm:", l1_norm(x))
print("L2 norm:", l2_norm(x))
print("L infinity norm:", linf_norm(x))

a = np.array([1, 2], dtype=float)
b = np.array([4, 6], dtype=float)

print()
print("Euclidean distance between a and b:", euclidean_distance(a, b))
print("Manhattan distance between a and b:", manhattan_distance(a, b))
print("Cosine similarity between a and b:", cosine_similarity(a, b))

X = np.array([
    [1.0, 100.0],
    [2.0, 130.0],
    [3.0, 170.0],
    [8.0, 700.0],
    [9.0, 740.0],
    [10.0, 780.0],
])

query = np.array([5.0, 260.0])

index_before, distance_before = nearest_neighbor(X, query)

print()
print("Nearest neighbor before scaling:")
print("index:", index_before)
print("distance:", distance_before)
print("point:", X[index_before])

X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
query_scaled = (query - X.mean(axis=0)) / X.std(axis=0)

index_after, distance_after = nearest_neighbor(X_scaled, query_scaled)

print()
print("Nearest neighbor after scaling:")
print("index:", index_after)
print("distance:", distance_after)
print("original point:", X[index_after])

print()
print("Pairwise Euclidean distance matrix:")
print(pairwise_euclidean_distances(X_scaled))

vectors = np.array([
    [1.0, 0.9],
    [0.9, 1.0],
    [-1.0, -0.8],
    [-0.8, -1.0],
    [0.1, 1.2],
])

cosine_matrix = np.zeros((vectors.shape[0], vectors.shape[0]))

for i in range(vectors.shape[0]):
    for j in range(vectors.shape[0]):
        cosine_matrix[i, j] = cosine_similarity(vectors[i], vectors[j])

print()
print("Cosine similarity matrix:")
print(cosine_matrix)
