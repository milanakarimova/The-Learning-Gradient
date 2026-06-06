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
