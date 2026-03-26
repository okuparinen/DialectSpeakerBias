from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import sys

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, alpha=0.05):
    """
    Compute bootstrap confidence interval.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = len(y_true)
    rng = np.random.default_rng()

    boot_scores = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        boot_scores[i] = metric_fn(y_true[idx], y_pred[idx])

    lower = np.percentile(boot_scores, 100 * (alpha / 2))
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))

    point = metric_fn(y_true, y_pred)
    return point, lower, upper

FOLDER = sys.argv[1]
TYPE = sys.argv[2]

base = f"SKN/speaker_partitions/{FOLDER}/{TYPE}/outputs"

dev_best = pd.read_csv(f"{base}/best_model/dev_predictions.csv")
test_best = pd.read_csv(f"{base}/best_model/test_predictions.csv")

datasets = {
    "dev_best": dev_best,
    "test_best": test_best,
}

print(f"{FOLDER}_{TYPE}")

for name, df in datasets.items():
    y_true = df["true_label"]
    y_pred = df["pred_label"]

    # Accuracy with CI
    acc, acc_low, acc_high = bootstrap_ci(
        y_true, y_pred, accuracy_score
    )

    # Macro F1 with CI
    f1, f1_low, f1_high = bootstrap_ci(
        y_true, y_pred, lambda yt, yp: f1_score(yt, yp, average="macro")
    )

    print(f"{name} accuracy: {acc:.4f} (95% CI: {acc_low:.4f} – {acc_high:.4f})")
    print(f"{name} macro F1: {f1:.4f} (95% CI: {f1_low:.4f} – {f1_high:.4f})\n")
