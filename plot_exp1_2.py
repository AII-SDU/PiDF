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

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# =========================
# Basic utilities
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
# Label and deduplication
# =========================
def build_label_column(df: pd.DataFrame, label_mode: str, ehull_threshold: float) -> pd.DataFrame:
    df = df.copy()

    if label_mode == "label":
        if "label" not in df.columns:
            raise ValueError("CSV is missing label column.")
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        return df

    if label_mode == "ehull":
        if "energy_above_hull" not in df.columns:
            raise ValueError("CSV is missing energy_above_hull column.")
        df = df[df["energy_above_hull"].notna()].copy()
        df["label"] = (df["energy_above_hull"].astype(float) <= ehull_threshold).astype(int)
        return df

    if label_mode == "is_stable":
        if "is_stable" not in df.columns:
            raise ValueError("CSV is missing is_stable column.")
        df = df[df["is_stable"].notna()].copy()
        df["label"] = df["is_stable"].astype(int)
        return df

    raise ValueError("label_mode must be label / ehull / is_stable")


def canonicalize_normalized_composition(df: pd.DataFrame, ndigits: int = 8) -> pd.DataFrame:
    df = df.copy()

    if "normalized_composition" not in df.columns:
        raise ValueError("CSV must contain normalized_composition for 254-composition deduplication.")

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
    before = len(df)

    if "energy_above_hull" in df.columns:
        df = df.sort_values("energy_above_hull", ascending=True, kind="mergesort")

    df = df.drop_duplicates(subset=["normalized_composition_key"], keep="first").reset_index(drop=True)

    after = len(df)
    print(f"[INFO] Composition deduplication: {before} -> {after}")

    return df


# =========================
# Feature engineering
# =========================
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


def prepare_dataset(csv_path: str, label_mode: str, ehull_threshold: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df = build_label_column(df, label_mode=label_mode, ehull_threshold=ehull_threshold)
    df = canonicalize_normalized_composition(df)
    df = deduplicate_by_composition(df)

    df = enrich_composition_columns(df)
    df = enrich_fraction_features(df)

    required_cols = ["classical_t", "wtf", "sigma_A", "sigma_B", "delta_q"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required descriptor columns: {missing}")

    df = df[df[required_cols].notna().all(axis=1)].copy().reset_index(drop=True)

    print("[INFO] Final dataset size:", len(df))
    print("[INFO] Stable:", int((df["label"] == 1).sum()))
    print("[INFO] Unstable:", int((df["label"] == 0).sum()))

    return df


# =========================
# Feature sets
# =========================
def _ordered_unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def get_feature_sets(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    frac_num = sorted([c for c in df.columns if c.startswith("frac_")])

    conventional_cat = [c for c in ["mix_site", "A_host", "A_sub", "B_host", "B_sub"] if c in df.columns]

    identity_num = [c for c in ["mix_ratio", "n_A_species", "n_B_species"] if c in df.columns]

    raw_num = [c for c in ["mix_ratio", "r_A_eff", "r_B_eff", "q_A_eff", "q_B_eff"] if c in df.columns]

    conventional_num = _ordered_unique(identity_num + frac_num + raw_num)

    feature_sets = {
        "Composition Vector + ML": {
            "categorical": [c for c in ["mix_site"] if c in df.columns],
            "numeric": _ordered_unique(identity_num + frac_num),
        },
        "Conventional Composition/Site + ML": {
            "categorical": conventional_cat,
            "numeric": conventional_num,
        },
        "Raw Radius/Valence + ML": {
            "categorical": [c for c in ["mix_site"] if c in df.columns],
            "numeric": raw_num,
        },
        "Classical TF + ML": {
            "categorical": conventional_cat,
            "numeric": _ordered_unique(conventional_num + [c for c in ["classical_t"] if c in df.columns]),
        },
        "Weighted TF + ML": {
            "categorical": conventional_cat,
            "numeric": _ordered_unique(conventional_num + [c for c in ["wtf"] if c in df.columns]),
        },
        "PiDF + ML": {
            "categorical": conventional_cat,
            "numeric": _ordered_unique(conventional_num + [c for c in ["wtf", "sigma_A", "sigma_B", "delta_q"] if c in df.columns]),
        },
    }

    return feature_sets


# =========================
# Model and PR evaluation
# =========================
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


def fit_predict_probs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    logreg_c: float,
) -> np.ndarray:
    used_cols = list(cat_cols) + list(num_cols)

    pre = build_preprocessor(cat_cols, num_cols)

    X_train = pre.fit_transform(train_df[used_cols])
    X_test = pre.transform(test_df[used_cols])

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    y_train = train_df["label"].values.astype(int)

    model = LogisticRegression(
        C=logreg_c,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    return probs


def compute_mean_pr(
    df: pd.DataFrame,
    feature_sets: Dict[str, Dict[str, List[str]]],
    seeds: List[int],
    test_size: float,
    logreg_c: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute mean precision-recall curves over random splits.

    Note:
    precision_recall_curve returns recall values in decreasing order in many
    sklearn versions. We reverse and sort before interpolation.
    """
    mean_recall = np.linspace(0, 1, 201)
    results = {}

    y = df["label"].values.astype(int)
    indices = np.arange(len(df))

    for method, spec in feature_sets.items():
        precisions_interp = []
        aps = []

        for seed in seeds:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=seed,
                stratify=y,
            )

            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            y_test = test_df["label"].values.astype(int)

            probs = fit_predict_probs(
                train_df=train_df,
                test_df=test_df,
                cat_cols=spec["categorical"],
                num_cols=spec["numeric"],
                logreg_c=logreg_c,
            )

            precision, recall, _ = precision_recall_curve(y_test, probs)
            ap = average_precision_score(y_test, probs)

            # Make recall increasing for interpolation
            order = np.argsort(recall)
            recall_sorted = recall[order]
            precision_sorted = precision[order]

            # Remove duplicate recall points by keeping the maximum precision
            recall_unique = []
            precision_unique = []
            for r in np.unique(recall_sorted):
                mask = recall_sorted == r
                recall_unique.append(r)
                precision_unique.append(np.max(precision_sorted[mask]))

            recall_unique = np.array(recall_unique)
            precision_unique = np.array(precision_unique)

            interp_precision = np.interp(mean_recall, recall_unique, precision_unique)
            precisions_interp.append(interp_precision)
            aps.append(ap)

        precisions_interp = np.array(precisions_interp)
        aps = np.array(aps)

        results[method] = {
            "mean_recall": mean_recall,
            "mean_precision": precisions_interp.mean(axis=0),
            "std_precision": precisions_interp.std(axis=0),
            "ap_mean": aps.mean(),
            "ap_std": aps.std(),
        }

    return results


# =========================
# Plotting
# =========================
def plot_mean_pr(
    pr_results: Dict[str, Dict[str, np.ndarray]],
    methods_to_plot: List[str],
    outdir: str,
    output_name: str,
) -> None:
    plt.figure(figsize=(6.2, 5.2))

    for method in methods_to_plot:
        if method not in pr_results:
            continue

        r = pr_results[method]
        recall = r["mean_recall"]
        precision = r["mean_precision"]
        std = r["std_precision"]

        label = f"{method} (AP={r['ap_mean']:.3f}$\\pm${r['ap_std']:.3f})"

        plt.plot(recall, precision, linewidth=1.8, label=label)
        plt.fill_between(
            recall,
            np.maximum(precision - std, 0),
            np.minimum(precision + std, 1),
            alpha=0.12,
        )

    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("PR-AUC")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11, loc="lower left")
    plt.tight_layout()

    png_path = os.path.join(outdir, f"{output_name}.png")
    pdf_path = os.path.join(outdir, f"{output_name}.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved: {png_path}")
    print(f"[INFO] Saved: {pdf_path}")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./mp_data/exp42_perovskite_like_subset.csv")
    parser.add_argument("--outdir", type=str, default="runs/pr_254")
    parser.add_argument("--label-mode", type=str, default="ehull", choices=["label", "ehull", "is_stable"])
    parser.add_argument("--ehull-threshold", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    args = parser.parse_args()

    ensure_dir(args.outdir)
    set_seed(0)

    df = prepare_dataset(
        csv_path=args.csv,
        label_mode=args.label_mode,
        ehull_threshold=args.ehull_threshold,
    )

    feature_sets = get_feature_sets(df)

    # 建议只画关键方法，图不会太乱
    methods_to_plot = [
        "Composition Vector + ML",
        "Raw Radius/Valence + ML",
        "Weighted TF + ML",
        "PiDF + ML",
    ]

    pr_results = compute_mean_pr(
        df=df,
        feature_sets=feature_sets,
        seeds=args.seeds,
        test_size=args.test_size,
        logreg_c=args.logreg_c,
    )

    # 保存数值结果
    summary_rows = []
    for method, r in pr_results.items():
        summary_rows.append({
            "method": method,
            "ap_mean": float(r["ap_mean"]),
            "ap_std": float(r["ap_std"]),
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.outdir, "mean_pr_summary.csv"),
        index=False,
    )

    plot_mean_pr(
        pr_results=pr_results,
        methods_to_plot=methods_to_plot,
        outdir=args.outdir,
        output_name="mean_pr_254",
    )


if __name__ == "__main__":
    main()