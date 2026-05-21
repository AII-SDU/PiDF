#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
INPUT_JSON = "runs/section3_logreg/section3_summary.json"
OUTPUT_DIR = "figures"
OUTPUT_NAME = "fig_ablation_delta_auc"

FAMILY = "incremental"
BASELINE_METHOD = "Conventional base"

METHOD_ORDER = [
    "Conventional base",
    "Base + wtf",
    "Base + wtf sigma",
    "Base + wtf deltaq",
    "Base + PiDF full",
]

LABEL_MAP = {
    "Conventional base": "Base\n(comp./site)",
    "Base + wtf": "Base + $t_w$",
    "Base + wtf sigma": "Base + $t_w+\\sigma$",
    "Base + wtf deltaq": "Base + $t_w+\\Delta q$",
    "Base + PiDF full": "Base + PiDF",
}

# # 哪个实验块
# FAMILY = "descriptor_only"

# # 以哪个方法作为 baseline
# BASELINE_METHOD = "wtf only"

# # 想展示的方法顺序
# METHOD_ORDER = [
#     "wtf only",
#     "wtf sigma",
#     "wtf deltaq",
#     "PiDF full",
# ]

# # 图中显示的标签
# LABEL_MAP = {
#     "wtf only": "$t_w$ only\n(geometry)",
#     "wtf sigma": "$t_w+\\sigma_A+\\sigma_B$\n(radius disorder)",
#     "wtf deltaq": "$t_w+\\Delta q$\n(charge mismatch)",
#     "PiDF full": "Full PiDF\n(all terms)",
# }


# 每个柱子的颜色（类似你给的那张图：不同柱子不同颜色）
COLOR_MAP = {
    "Conventional base": "#C9C9C9",   # 灰色
    "Base + wtf": "#F5A623",          # 橙色
    "Base + wtf sigma": "#F5A623",    # 橙色
    "Base + wtf deltaq": "#8E63CE",   # 紫色
    "Base + PiDF full": "#2E73B8",    # 蓝色
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_auc(summary, family, method):
    return float(summary["results"][family][method]["roc_auc"]["mean"])


def main():
    ensure_dir(OUTPUT_DIR)

    summary = load_summary(INPUT_JSON)

    baseline_auc = get_auc(summary, FAMILY, BASELINE_METHOD)

    delta_auc = []
    labels = []
    colors = []

    for method in METHOD_ORDER:
        auc = get_auc(summary, FAMILY, method)
        delta = auc - baseline_auc

        delta_auc.append(delta)
        labels.append(LABEL_MAP.get(method, method))
        colors.append(COLOR_MAP.get(method, "#4C78A8"))

    x = np.arange(len(METHOD_ORDER))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    bars = ax.bar(
        x,
        delta_auc,
        width=0.56,
        color=colors,
        edgecolor="none"
    )

    # 强调最后一个 All (PiDF)
    bars[-1].set_edgecolor("black")
    bars[-1].set_linewidth(1.5)

    # 零线
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.2)

    # 数值标注
    for i, val in enumerate(delta_auc):
        if val >= 0:
            y_text = val + 0.006
            va = "bottom"
        else:
            y_text = val - 0.006
            va = "top"

        if abs(val) < 5e-4:
            txt = "0.00"
        else:
            txt = f"{val:+.4f}"

        ax.text(
            x[i],
            y_text,
            txt,
            ha="center",
            va=va,
            fontsize=11,
            fontweight="bold" if i == len(delta_auc) - 1 else "normal"
        )

    # 坐标轴设置
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"$\Delta$ AUC (vs. Conventional base)", fontsize=12)
    # ax.set_title("Incremental Ablation Study ($\\Delta$AUC)", fontsize=14, fontweight="bold")

    # 去掉上右边框，更像论文示意图
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 纵轴范围可根据你的结果微调
    ymin = min(-0.01, min(delta_auc))
    ymax = max(0.03, max(delta_auc) + 0.01)
    ax.set_ylim(ymin, ymax)

    # 让版式更紧凑
    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME + ".png")
    pdf_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME + ".pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Baseline = {BASELINE_METHOD}, AUC = {baseline_auc:.4f}")


if __name__ == "__main__":
    main()