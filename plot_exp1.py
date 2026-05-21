import re
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# User settings
# =========================================================
INPUT_JSON = "runs/section1_logreg/benchmark_summary.json"
OUTPUT_DIR = "runs/section1_logreg/figures"

METHOD_ORDER = [
    "Classical TF Threshold",
    "Composition Vector + ML",
    "Raw Radius/Valence + ML",
    "Conventional Composition/Site + ML",
    "Classical TF + ML",
    "Weighted TF + ML",
    "PiDF + ML",
]

LABEL_MAP = {
    "Classical TF Threshold": "Classical TF Threshold",
    "Composition Vector + ML": "Composition Vector + ML",
    "Raw Radius/Valence + ML": "Raw Radius/Valence + ML",
    "Conventional Composition/Site + ML": "Conventional Comp./Site + ML",
    "Classical TF + ML": "Classical TF + ML",
    "Weighted TF + ML": "Weighted TF + ML",
    "PiDF + ML": "PiDF + ML",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric(summary, metric):
    rows = []
    for method in METHOD_ORDER:
        if method not in summary:
            continue
        if metric not in summary[method]:
            continue

        mean = summary[method][metric].get("mean", np.nan)
        std = summary[method][metric].get("std", np.nan)

        if mean is None or np.isnan(mean):
            continue

        rows.append({
            "method": method,
            "label": LABEL_MAP.get(method, method),
            "mean": float(mean),
            "std": float(std) if std is not None else 0.0,
        })
    return rows


def plot_metric_point(rows, metric_label, title, output_stem, xlim=None):
    """
    Horizontal point plot with mean ± std.
    """
    # reverse order so the first method appears at the top
    rows = rows[::-1]

    labels = [r["label"] for r in rows]
    means = np.array([r["mean"] for r in rows], dtype=float)
    stds = np.array([r["std"] for r in rows], dtype=float)
    methods = [r["method"] for r in rows]

    y = np.arange(len(rows))

    fig_height = max(4.2, 0.52 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))

    ax.errorbar(
        means,
        y,
        xerr=stds,
        fmt="o",
        capsize=4,
        markersize=6,
        linewidth=1.5
    )

    # Mark PiDF more clearly without changing the whole color scheme
    for yi, m, val in zip(y, methods, means):
        if m == "PiDF + ML":
            ax.plot(val, yi, marker="o", markersize=9, markeredgewidth=1.8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric_label)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        xmin = max(0.0, np.nanmin(means - stds) - 0.03)
        xmax = min(1.0, np.nanmax(means + stds) + 0.03)
        ax.set_xlim(xmin, xmax)

    # Bold the PiDF tick label
    for tick, m in zip(ax.get_yticklabels(), methods):
        if m == "PiDF + ML":
            tick.set_fontweight("bold")

    fig.tight_layout()

    pdf_path = os.path.join(OUTPUT_DIR, f"{output_stem}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{output_stem}.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def main():
    ensure_dir(OUTPUT_DIR)
    summary = load_summary(INPUT_JSON)

    # PR-AUC: threshold baseline has no probability output, so it is automatically omitted
    pr_rows = extract_metric(summary, "pr_auc")
    plot_metric_point(
        pr_rows,
        metric_label="PR-AUC",
        title="Benchmark Comparison: PR-AUC",
        output_stem="fig1a_pr_auc_point",
        xlim=(0.78, 0.98)
    )

    # Balanced accuracy: include the threshold baseline
    bal_rows = extract_metric(summary, "balanced_acc")
    plot_metric_point(
        bal_rows,
        metric_label="Balanced Accuracy",
        title="Benchmark Comparison: Balanced Accuracy",
        output_stem="fig1b_balanced_accuracy_point",
        xlim=(0.45, 0.86)
    )


if __name__ == "__main__":
    main()