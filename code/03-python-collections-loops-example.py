experiments = [
    {"model": "KNN", "accuracy": 0.82},
    {"model": "Logistic Regression", "accuracy": 0.86},
    {"model": "Random Forest", "accuracy": 0.89},
    {"model": "SVM", "accuracy": 0.84},
]

print("Models with accuracy greater than 0.85:")

for experiment in experiments:
    if experiment["accuracy"] > 0.85:
        print("-", experiment["model"], "with accuracy", experiment["accuracy"])

best_experiment = experiments[0]

for experiment in experiments:
    if experiment["accuracy"] > best_experiment["accuracy"]:
        best_experiment = experiment

print()
print("Best experiment:")
print("Model:", best_experiment["model"])
print("Accuracy:", best_experiment["accuracy"])

true_labels = [1, 0, 1, 1, 0]
predicted_labels = [1, 0, 0, 1, 0]

correct = 0

for i in range(len(true_labels)):
    if true_labels[i] == predicted_labels[i]:
        correct = correct + 1

accuracy = correct / len(true_labels)

print()
print("Manual accuracy example:")
print("Correct predictions:", correct)
print("Accuracy:", accuracy)
