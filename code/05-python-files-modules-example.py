from pathlib import Path

from ml_metrics import calculate_accuracy, calculate_mean, calculate_mse


true_labels = [1, 0, 1, 1, 0]
predicted_labels = [1, 0, 0, 1, 0]

accuracy = calculate_accuracy(true_labels, predicted_labels)

y_true = [3, 5, 2, 7]
y_pred = [2.5, 5.5, 2, 8]

mse = calculate_mse(y_true, y_pred)

experiment_scores = [0.72, 0.75, 0.80, 0.82]
mean_score = calculate_mean(experiment_scores)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

summary_path = output_dir / "experiment_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("Experiment Summary\n")
    file.write("==================\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"MSE: {mse}\n")
    file.write(f"Mean score: {mean_score}\n")

print("Experiment summary saved to:", summary_path)
print("Accuracy:", accuracy)
print("MSE:", mse)
print("Mean score:", mean_score)
