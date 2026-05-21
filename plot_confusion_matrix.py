#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import json
import random
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# =========================
# Utilities
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_list_like(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        try:
            out = json.loads(s)
            if isinstance(out, list):
                return out
        except Exception:
            pass
        try:
            out = ast.literal_eval(s)
            if isinstance(out, (list, tuple)):
                return list(out)
        except Exception:
            pass
    return []


def _parse_dict_like(x: Any) -> Dict[str, float]:
    if isinstance(x, dict):
        return {str(k): float(v) for k, v in x.items()}
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return {}
        try:
            out = json.loads(s)
            if isinstance(out, dict):
                return {str(k): float(v) for k, v in out.items()}
        except Exception:
            pass
        try:
            out = ast.literal_eval(s)
            if isinstance(out, dict):
                return {str(k): float(v) for k, v in out.items()}
        except Exception:
            pass
    return {}


def _safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


# =========================
# Dataset preparation
# =========================
def build_label_column(df: pd.DataFrame, ehull_threshold: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    if "energy_above_hull" not in df.columns:
        raise ValueError("CSV must contain 'energy_above_hull' column.")
    df = df[df["energy_above_hull"].notna()].copy()
    df["label"] = (df["energy_above_hull"].astype(float) <= ehull_threshold).astype(int)
    return df


def canonicalize_normalized_composition(df: pd.DataFrame, ndigits: int = 8) -> pd.DataFrame:
    df = df.copy()

    if "normalized_composition" not in df.columns:
        raise ValueError("CSV must contain 'normalized_composition' column.")

    keys = []
    for x in df["normalized_composition"].tolist():
        d = _parse_dict_like(x)
        items = sorted(
            [(str(k), round(float(v), ndigits)) for k, v in d.items()],
            key=lambda kv: kv[0],
        )
        keys.append(json.dumps(items, ensure_ascii=False))

    df["normalized_composition_key"] = keys
    return df


def deduplicate_by_composition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "energy_above_hull" in df.columns:
        df = df.sort_values("energy_above_hull", ascending=True, kind="mergesort")

    before = len(df)
    df = df.drop_duplicates(subset=["normalized_composition_key"], keep="first").reset_index(drop=True)
    after = len(df)

    print(f"[INFO] Composition deduplication: {before} -> {after}")
    return df


def enrich_composition_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    needed = {"A_elements", "A_amounts", "B_elements", "B_amounts"}
    if not needed.issubset(df.columns):
        return df

    records = []

    for _, row in df.iterrows():
        mix_site = row.get("mix_site", None)

        A_elements = [str(v) for v in _parse_list_like(row.get("A_elements"))]
        A_amounts = [_safe_float(v) for v in _parse_list_like(row.get("A_amounts"))]
        B_elements = [str(v) for v in _parse_list_like(row.get("B_elements"))]
        B_amounts = [_safe_float(v) for v in _parse_list_like(row.get("B_amounts"))]

        rec = {
            "A_host": None,
            "A_sub": None,
            "B_host": None,
            "B_sub": None,
            "mix_ratio": row.get("mix_ratio", None),
            "n_A_species": len(A_elements),
            "n_B_species": len(B_elements),
        }

        def choose_host_sub(elements, amounts):
            pairs = []
            for el, amt in zip(elements, amounts):
                if amt is None:
                    continue
                pairs.append((el, float(amt)))
            if len(pairs) == 0:
                return None, None, None
            pairs = sorted(pairs, key=lambda t: (-t[1], t[0]))
            host = pairs[0][0]
            if len(pairs) == 1:
                return host, None, 0.0
            sub = pairs[1][0]
            ratio = float(sorted([pairs[0][1], pairs[1][1]])[0])
            return host, sub, ratio

        if mix_site == "A":
            A_host, A_sub, ratio = choose_host_sub(A_elements, A_amounts)
            rec["A_host"] = A_host
            rec["A_sub"] = A_sub
            rec["mix_ratio"] = ratio if ratio is not None else rec["mix_ratio"]

            if len(B_elements) >= 1:
                B_sorted = sorted(B_elements)
                rec["B_host"] = B_sorted[0]
                rec["B_sub"] = B_sorted[1] if len(B_sorted) > 1 else None

        elif mix_site == "B":
            B_host, B_sub, ratio = choose_host_sub(B_elements, B_amounts)
            rec["B_host"] = B_host
            rec["B_sub"] = B_sub
            rec["mix_ratio"] = ratio if ratio is not None else rec["mix_ratio"]

            if len(A_elements) >= 1:
                A_sorted = sorted(A_elements)
                rec["A_host"] = A_sorted[0]
                rec["A_sub"] = A_sorted[1] if len(A_sorted) > 1 else None

        records.append(rec)

    derived = pd.DataFrame(records, index=df.index)
    for col in derived.columns:
        if col not in df.columns:
            df[col] = derived[col]
        else:
            df[col] = df[col].where(df[col].notna(), derived[col])

    return df


def enrich_fraction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "normalized_composition" not in df.columns:
        return df

    parsed = []
    elements = set()

    for x in df["normalized_composition"].tolist():
        d = _parse_dict_like(x)
        parsed.append(d)
        for k in d.keys():
            if str(k) != "O":
                elements.add(str(k))

    elements = sorted(elements)

    for el in elements:
        df[f"frac_{el}"] = [float(d.get(el, 0.0)) for d in parsed]

    return df


def prepare_dataset(csv_path: str, ehull_threshold: float = 0.05) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = build_label_column(df, ehull_threshold=ehull_threshold)
    df = canonicalize_normalized_composition(df)
    df = deduplicate_by_composition(df)
    df = enrich_composition_columns(df)
    df = enrich_fraction_features(df)

    required_cols = ["wtf", "sigma_A", "sigma_B", "delta_q"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required PiDF columns: {missing}")

    df = df[df[required_cols].notna().all(axis=1)].copy().reset_index(drop=True)

    print("[INFO] Final dataset size:", len(df))
    print("[INFO] Near-stable:", int((df["label"] == 1).sum()))
    print("[INFO] Unstable:", int((df["label"] == 0).sum()))

    return df


# =========================
# PiDF feature set
# =========================
def _ordered_unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def get_pidf_feature_spec(df: pd.DataFrame) -> Dict[str, List[str]]:
    frac_num = sorted([c for c in df.columns if c.startswith("frac_")])

    cat_cols = [c for c in ["mix_site", "A_host", "A_sub", "B_host", "B_sub"] if c in df.columns]

    base_num = [c for c in ["mix_ratio", "n_A_species", "n_B_species"] if c in df.columns]
    raw_num = [c for c in ["r_A_eff", "r_B_eff", "q_A_eff", "q_B_eff"] if c in df.columns]

    num_cols = _ordered_unique(
        base_num + frac_num + raw_num + [c for c in ["wtf", "sigma_A", "sigma_B", "delta_q"] if c in df.columns]
    )

    return {
        "categorical": cat_cols,
        "numeric": num_cols,
    }


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    transformers = []

    if len(cat_cols) > 0:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if len(num_cols) > 0:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    C: float = 1.0,
    threshold: float = 0.5,
):
    used_cols = list(cat_cols) + list(num_cols)

    pre = build_preprocessor(cat_cols, num_cols)
    X_train = pre.fit_transform(train_df[used_cols])
    X_test = pre.transform(test_df[used_cols])

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    y_train = train_df["label"].values.astype(int)

    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    return probs, preds


# =========================
# Plot confusion matrix
# =========================
def plot_styled_confusion_matrix(
    cm: np.ndarray,
    acc: float,
    save_png: str,
    save_pdf: str,
    title: str = "PiDF",
):
    # cm layout:
    # [[TP, FN],
    #  [FP, TN]]
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]

    fig, ax = plt.subplots(figsize=(5.3, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Outer rounded box
    outer = FancyBboxPatch(
        (0.02, 0.04), 0.96, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2, edgecolor="#BDBDBD", facecolor="white"
    )
    ax.add_patch(outer)

    # Title
    ax.text(0.5, 0.91, title, ha="center", va="center", fontsize=15, fontweight="bold")

    # Column labels
    ax.text(0.42, 0.84, "Pred. Near-stable", ha="center", va="center", fontsize=14, color="#333333")
    ax.text(0.74, 0.84, "Pred. Unstable", ha="center", va="center", fontsize=14, color="#333333")

    # Cell block geometry
    x0 = 0.30
    y0 = 0.34
    w = 0.32
    h = 0.20

    # Row label area
    ax.add_patch(Rectangle((0.05, y0), 0.25, 2*h, facecolor="#F3F3F3", edgecolor="#D0D0D0", linewidth=0.8))
    ax.add_patch(Rectangle((0.05, y0+h), 0.25, h, facecolor="#F3F3F3", edgecolor="#D0D0D0", linewidth=0.8))
    ax.add_patch(Rectangle((0.05, y0), 0.25, h, facecolor="#F3F3F3", edgecolor="#D0D0D0", linewidth=0.8))

    ax.text(0.175, y0 + 1.5*h, "True\nNear-stable", ha="center", va="center", fontsize=14, color="#333333")
    ax.text(0.175, y0 + 0.5*h, "True\nUnstable", ha="center", va="center", fontsize=14, color="#333333")

    # 2x2 colored cells
    green = "#DDECD6"
    red = "#F4DEDF"
    edge = "#C6C6C6"

    # Top-left: TP
    ax.add_patch(Rectangle((x0, y0+h), w, h, facecolor=green, edgecolor=edge, linewidth=0.9))
    # Top-right: FN
    ax.add_patch(Rectangle((x0+w, y0+h), w, h, facecolor=red, edgecolor=edge, linewidth=0.9))
    # Bottom-left: FP
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=red, edgecolor=edge, linewidth=0.9))
    # Bottom-right: TN
    ax.add_patch(Rectangle((x0+w, y0), w, h, facecolor=green, edgecolor=edge, linewidth=0.9))

    # Numbers
    ax.text(x0 + w/2, y0 + 1.5*h, f"{tp}", ha="center", va="center", fontsize=18, color="#222222")
    ax.text(x0 + 1.5*w, y0 + 1.5*h, f"{fn}", ha="center", va="center", fontsize=18, color="#222222")
    ax.text(x0 + w/2, y0 + 0.5*h, f"{fp}", ha="center", va="center", fontsize=18, color="#222222")
    ax.text(x0 + 1.5*w, y0 + 0.5*h, f"{tn}", ha="center", va="center", fontsize=18, color="#222222")

    # Accuracy box at bottom
    acc_box = FancyBboxPatch(
        (0.05, 0.10), 0.90, 0.12,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        linewidth=1.0, edgecolor="#BDBDBD", facecolor="white"
    )
    ax.add_patch(acc_box)
    ax.text(0.5, 0.16, f"Acc = {acc:.3f}", ha="center", va="center", fontsize=16, color="#333333")

    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved: {save_png}")
    print(f"[INFO] Saved: {save_pdf}")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./mp_data/exp42_perovskite_like_subset.csv")
    parser.add_argument("--outdir", type=str, default="runs/confusion_matrix_254")
    parser.add_argument("--ehull-threshold", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    set_seed(args.seed)

    df = prepare_dataset(args.csv, ehull_threshold=args.ehull_threshold)
    feat = get_pidf_feature_spec(df)

    indices = np.arange(len(df))
    y = df["label"].values.astype(int)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    probs, preds = fit_predict(
        train_df=train_df,
        test_df=test_df,
        cat_cols=feat["categorical"],
        num_cols=feat["numeric"],
        C=args.logreg_c,
        threshold=args.threshold,
    )

    y_true = test_df["label"].values.astype(int)

    # labels=[1,0] ensures:
    # row 0 = true near-stable
    # row 1 = true unstable
    # col 0 = pred near-stable
    # col 1 = pred unstable
    cm = confusion_matrix(y_true, preds, labels=[1, 0])
    acc = accuracy_score(y_true, preds)

    print("[INFO] Confusion matrix (rows=true, cols=pred, order=[near-stable, unstable])")
    print(cm)
    print(f"[INFO] Accuracy = {acc:.4f}")

    pd.DataFrame(
        cm,
        index=["True Near-stable", "True Unstable"],
        columns=["Pred Near-stable", "Pred Unstable"]
    ).to_csv(os.path.join(args.outdir, "confusion_matrix_values.csv"))

    with open(os.path.join(args.outdir, "confusion_matrix_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "test_size": args.test_size,
                "threshold": args.threshold,
                "accuracy": float(acc),
                "confusion_matrix": cm.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_styled_confusion_matrix(
        cm=cm,
        acc=acc,
        save_png=os.path.join(args.outdir, "confusion_matrix_pidf_254.png"),
        save_pdf=os.path.join(args.outdir, "confusion_matrix_pidf_254.pdf"),
        title="PiDF",
    )


if __name__ == "__main__":
    main()