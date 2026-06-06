def calculate_mean(values):
    return sum(values) / len(values)


def calculate_accuracy(true_labels, predicted_labels):
    correct = 0

    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            correct = correct + 1

    return correct / len(true_labels)


def calculate_mse(y_true, y_pred):
    total_error = 0

    for i in range(len(y_true)):
        error = y_true[i] - y_pred[i]
        squared_error = error ** 2
        total_error = total_error + squared_error

    return total_error / len(y_true)


scores = [0.72, 0.75, 0.80, 0.82]
mean_score = calculate_mean(scores)

print("Mean score:", mean_score)

true_labels = [1, 0, 1, 1, 0]
predicted_labels = [1, 0, 0, 1, 0]

accuracy = calculate_accuracy(true_labels, predicted_labels)

print("Accuracy:", accuracy)

y_true = [3, 5, 2, 7]
y_pred = [2.5, 5.5, 2, 8]

mse = calculate_mse(y_true, y_pred)

print("MSE:", mse)
