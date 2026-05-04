"""Generate result figures from the aggregate experiment CSV."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cache"))

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("results/aggregated.csv")
DEFAULT_FIGURE_DIR = Path("results/figures")


def read_aggregate_rows(path: Path | str) -> list[dict[str, Any]]:
    """Read aggregate CSV rows and coerce numeric fields used for plotting."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["train_size"] = int(row["train_size"])
            row["budget"] = int(row["budget"])
            row["accuracy"] = float(row["accuracy"])
            row["answer_format_ok_rate"] = float(row["answer_format_ok_rate"])
            row["train_cost_usd"] = float(row["train_cost_usd"])
            row["inference_cost_per_query_usd"] = float(
                row["inference_cost_per_query_usd"]
            )
            rows.append(row)
    if not rows:
        raise ValueError(f"No aggregate rows found in {path}")
    return rows


def strategy_label(row: dict[str, Any]) -> str:
    """Return a compact display label for one strategy row."""

    if row["strategy"] == "greedy":
        return "Greedy"
    if row["strategy"] == "sc":
        return f"SC@{row['budget']}"
    return f"{row['strategy']}@{row['budget']}"


def build_metric_matrix(
    rows: list[dict[str, Any]],
    metric: str,
    train_sizes: list[int],
    strategy_labels: list[str],
) -> list[list[float]]:
    """Build a train_size x strategy matrix for heatmaps."""

    by_key = {
        (row["train_size"], strategy_label(row)): float(row[metric]) for row in rows
    }
    matrix = []
    for train_size in train_sizes:
        matrix.append([by_key[(train_size, label)] for label in strategy_labels])
    return matrix


def plot_accuracy_heatmap(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot accuracy by train size and inference strategy."""

    train_sizes = sorted({row["train_size"] for row in rows})
    labels = ["Greedy", "SC@4", "SC@8"]
    matrix = build_metric_matrix(rows, "accuracy", train_sizes, labels)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(train_sizes)), labels=[str(size) for size in train_sizes])
    ax.set_xlabel("Inference strategy")
    ax.set_ylabel("Synthetic train size")
    ax.set_title("GSM8K Accuracy by Training and Inference-Time Scaling")

    for row_index, train_size in enumerate(train_sizes):
        for col_index, _label in enumerate(labels):
            value = matrix[row_index][col_index]
            color = "white" if value >= 0.65 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontweight="bold",
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Exact-match accuracy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cost_accuracy(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot accuracy against current cost accounting columns."""

    labels = ["Greedy", "SC@4", "SC@8"]
    colors = {"Greedy": "#3B82F6", "SC@4": "#10B981", "SC@8": "#F59E0B"}
    markers = {0: "o", 100: "s", 500: "^"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in labels:
        label_rows = sorted(
            [row for row in rows if strategy_label(row) == label],
            key=lambda row: row["train_cost_usd"],
        )
        xs = [row["train_cost_usd"] for row in label_rows]
        ys = [row["accuracy"] for row in label_rows]
        ax.plot(xs, ys, color=colors[label], linewidth=1.8, label=label)
        for row in label_rows:
            train_size = row["train_size"]
            ax.scatter(
                row["train_cost_usd"],
                row["accuracy"],
                color=colors[label],
                marker=markers[train_size],
                s=90,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            ax.annotate(
                f"n={train_size}",
                (row["train_cost_usd"], row["accuracy"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

    ax.set_xlabel("One-time training cost, USD")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0.3, 0.82)
    ax.set_title("Accuracy vs Training Cost\nserving-token cost not yet included")
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(title="Strategy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_aggregate_rows(args.input)
    plot_accuracy_heatmap(rows, args.output_dir / "accuracy_heatmap.png")
    plot_cost_accuracy(rows, args.output_dir / "cost_accuracy_training_only.png")
    print(f"Wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
