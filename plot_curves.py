"""Plot training loss and test accuracy for every run under results/.

Example:
    python plot_curves.py            # writes results/curves.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path: Path) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                columns.setdefault(key, []).append(float(value))
    return columns


def main() -> None:
    runs = sorted(Path("results").glob("*/metrics.csv"))
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4))
    for path in runs:
        metrics = read_metrics(path)
        name = path.parent.name
        loss_ax.plot(metrics["epoch"], metrics["train_loss"], label=name)
        acc_ax.plot(metrics["epoch"], [a * 100 for a in metrics["test_accuracy"]], label=name)

    loss_ax.set_title("training loss")
    acc_ax.set_title("test accuracy (%)")
    for ax in (loss_ax, acc_ax):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    acc_ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("results/curves.png", dpi=150)
    print(f"wrote results/curves.png from {len(runs)} runs")


if __name__ == "__main__":
    main()
