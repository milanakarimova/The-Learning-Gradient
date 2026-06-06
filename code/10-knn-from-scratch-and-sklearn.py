from collections import Counter

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def get_k_nearest_neighbors(X_train, query_point, k):
    distances = []

    for index, train_point in enumerate(X_train):
        distance = euclidean_distance(train_point, query_point)
        distances.append((distance, index))

    distances.sort(key=lambda item: item[0])

    return distances[:k]


def knn_predict_classification(X_train, y_train, query_point, k):
    neighbors = get_k_nearest_neighbors(X_train, query_point, k)

    neighbor_labels = []

    for distance, index in neighbors:
        neighbor_labels.append(y_train[index])

    vote_counts = Counter(neighbor_labels)
    prediction = vote_counts.most_common(1)[0][0]

    return prediction


print("KNN from scratch")
print("================")

X_toy = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7],
    [8, 6],
])

y_toy = np.array([0, 0, 0, 1, 1, 1])

query_point = np.array([4, 4])

prediction = knn_predict_classification(
    X_toy,
    y_toy,
    query_point,
    k=3
)

print("Query point:", query_point)
print("Prediction:", prediction)

print()
print("KNN with Scikit-learn on Iris")
print("============================")

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for k in [1, 3, 5, 7, 9, 11]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"k={k}, test accuracy={accuracy:.3f}")
