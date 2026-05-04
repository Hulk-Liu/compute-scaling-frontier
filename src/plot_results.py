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
from matplotlib.lines import Line2D


DEFAULT_INPUT = Path("results/aggregated.csv")
DEFAULT_FIGURE_DIR = Path("results/figures")
QUERY_COUNT_COLUMNS = (
    ("1K", "total_cost_usd_at_1000"),
    ("10K", "total_cost_usd_at_10000"),
    ("100K", "total_cost_usd_at_100000"),
    ("1M", "total_cost_usd_at_1000000"),
)
STRATEGY_ORDER = ("Greedy", "SC@4", "SC@8", "BoN@4")
STRATEGY_COLORS = {
    "Greedy": "#3B82F6",
    "SC@4": "#10B981",
    "SC@8": "#F59E0B",
    "BoN@4": "#8B5CF6",
}
TRAIN_SIZE_MARKERS = {0: "o", 100: "s", 500: "^"}


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
            for _label, column in QUERY_COUNT_COLUMNS:
                if column in row:
                    row[column] = float(row[column])
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
    if row["strategy"] == "bon":
        return f"BoN@{row['budget']}"
    return f"{row['strategy']}@{row['budget']}"


def ordered_strategy_labels(rows: list[dict[str, Any]]) -> list[str]:
    """Return strategy labels in the preferred display order."""

    labels = {strategy_label(row) for row in rows}
    ordered = [label for label in STRATEGY_ORDER if label in labels]
    ordered.extend(sorted(labels - set(ordered)))
    return ordered


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
    labels = ordered_strategy_labels(rows)
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

    labels = ordered_strategy_labels(rows)
    colors = {label: STRATEGY_COLORS[label] for label in labels}
    markers = TRAIN_SIZE_MARKERS

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in labels:
        for row in [row for row in rows if strategy_label(row) == label]:
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
    ax.set_title("Accuracy vs One-Time Training Cost\ntraining-only diagnostic")
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(handles=strategy_legend_handles(colors), title="Strategy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def strategy_legend_handles(colors: dict[str, str]) -> list[Line2D]:
    """Build legend handles for inference strategy colors."""

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
            label=label,
        )
        for label, color in colors.items()
    ]


def train_size_legend_handles(markers: dict[int, str]) -> list[Line2D]:
    """Build legend handles for train-size marker shapes."""

    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#E5E7EB",
            markeredgecolor="black",
            markersize=8,
            label=f"n={train_size}",
        )
        for train_size, marker in markers.items()
    ]


def plot_cost_accuracy_by_volume(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot accuracy against total cost at several query volumes."""

    labels = ordered_strategy_labels(rows)
    colors = {label: STRATEGY_COLORS[label] for label in labels}
    markers = TRAIN_SIZE_MARKERS

    fig, axes = plt.subplots(
        len(QUERY_COUNT_COLUMNS),
        1,
        figsize=(8.5, 11),
        sharey=True,
    )
    for ax, (volume_label, cost_column) in zip(axes, QUERY_COUNT_COLUMNS):
        for label in labels:
            for row in [row for row in rows if strategy_label(row) == label]:
                train_size = row["train_size"]
                ax.scatter(
                    row[cost_column],
                    row["accuracy"],
                    color=colors[label],
                    marker=markers[train_size],
                    s=72,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )

        ax.set_title(f"{volume_label} queries", loc="left", fontweight="bold")
        ax.set_xlabel("Total cost, USD")
        ax.set_ylabel("Exact-match accuracy")
        ax.set_ylim(0.3, 0.82)
        ax.grid(True, axis="both", alpha=0.25)

    fig.legend(
        handles=strategy_legend_handles(colors),
        title="Strategy",
        loc="upper left",
        ncol=3,
        bbox_to_anchor=(0.12, 1.01),
    )
    fig.legend(
        handles=train_size_legend_handles(markers),
        title="Train size",
        loc="upper right",
        ncol=3,
        bbox_to_anchor=(0.88, 1.01),
    )
    fig.suptitle("Accuracy vs Total Cost by Query Volume", y=1.055, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
    plot_cost_accuracy_by_volume(rows, args.output_dir / "cost_accuracy_by_volume.png")
    print(f"Wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
