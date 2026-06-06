from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


output_dir = Path("outputs") / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

results = pd.DataFrame({
    "model": ["KNN", "LogReg", "RF", "SVM"],
    "accuracy": [0.82, 0.86, 0.89, 0.84],
    "training_time": [0.2, 0.5, 1.4, 0.8],
})

plt.figure()
plt.bar(results["model"], results["accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(output_dir / "model_accuracy_bar_chart.png", dpi=150)
plt.close()

epochs = np.arange(1, 11)
losses = np.array([0.95, 0.82, 0.70, 0.61, 0.54, 0.48, 0.43, 0.39, 0.36, 0.34])

plt.figure()
plt.plot(epochs, losses, marker="o")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "training_loss_line_plot.png", dpi=150)
plt.close()

rng = np.random.default_rng(seed=42)
errors = rng.normal(loc=0, scale=1, size=200)

plt.figure()
plt.hist(errors, bins=20)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(output_dir / "prediction_errors_histogram.png", dpi=150)
plt.close()

plt.figure()
plt.scatter(results["training_time"], results["accuracy"])
plt.title("Accuracy vs Training Time")
plt.xlabel("Training Time")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "accuracy_vs_training_time_scatter.png", dpi=150)
plt.close()

print("Figures saved to:", output_dir)
