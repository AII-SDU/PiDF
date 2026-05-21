#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Section 5: Prioritization of the Unlabeled Candidate Pool and Chemical Sanity Checks

This script trains a representation-level model on the labeled multicomponent oxide ABO3
benchmark and applies it to an unlabeled candidate pool. It then performs candidate-space
compression, family-level aggregation, and chemical sanity checks.

Main outputs:
- labeled_internal_validation.json
- scored_pool_raw.csv
- scored_pool_deduplicated.csv
- top_overall_candidates.csv
- top_chemically_conservative_candidates.csv
- top_charge_balanced_candidates.csv
- family_summary.csv
- family_concentration.json
- score_vs_mixratio_top_families.csv
- section5_summary.json
- plots/*.png and plots/*.pdf for the main score-distribution plot
"""

import os
import json
import ast
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# Utilities
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_json_loads(x: Any) -> Any:
    if isinstance(x, (list, dict)):
        return x
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def canonicalize_norm_comp(x: Any, decimals: int = 6) -> Optional[str]:
    d = _safe_json_loads(x)
    if not isinstance(d, dict):
        return None
    out = []
    for k, v in d.items():
        try:
            vv = round(float(v), decimals)
        except Exception:
            return None
        out.append((str(k), vv))
    out.sort(key=lambda t: t[0])
    return json.dumps(out, ensure_ascii=False)


def choose_dedup_key(df: pd.DataFrame, dedup_key: str) -> str:
    """
    Choose a composition-level deduplication key.

    For candidate prioritization, normalized_composition_key is preferred when
    available because formula strings may encode the same composition in
    different textual forms.
    """
    if dedup_key != "auto":
        if dedup_key not in df.columns:
            raise ValueError(f"Specified dedup key '{dedup_key}' not found in columns.")
        return dedup_key

    candidate_keys = [
        "normalized_composition_key",
        "candidate_formula_std",
        "candidate_formula_for_mp",
        "candidate_formula",
        "formula_pretty",
        "material_id",
    ]
    for key in candidate_keys:
        if key in df.columns and df[key].notna().any():
            return key
    raise ValueError(
        "Cannot determine dedup key automatically. "
        f"Available columns: {list(df.columns)}"
    )


def ensure_normalized_composition_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "normalized_composition_key" in df.columns and df["normalized_composition_key"].notna().any():
        return df
    if "normalized_composition" in df.columns:
        df["normalized_composition_key"] = df["normalized_composition"].apply(canonicalize_norm_comp)
        return df
    if "composition_dict" in df.columns:
        df["normalized_composition_key"] = df["composition_dict"].apply(canonicalize_norm_comp)
        return df
    for col in ["candidate_formula_std", "candidate_formula_for_mp", "candidate_formula"]:
        if col in df.columns:
            df["normalized_composition_key"] = df[col].astype(str)
            return df
    return df


def filter_self_substitutions(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    df = df.copy()
    required = {"mix_site", "A_host", "A_sub", "B_host", "B_sub"}
    if not required.issubset(df.columns):
        return df, 0
    mask_same_A = (df["mix_site"].astype(str) == "A") & (df["A_host"].astype(str) == df["A_sub"].astype(str))
    mask_same_B = (df["mix_site"].astype(str) == "B") & (df["B_host"].astype(str) == df["B_sub"].astype(str))
    mask = mask_same_A | mask_same_B
    removed = int(mask.sum())
    if removed > 0:
        df = df.loc[~mask].copy()
    return df, removed


def build_label_column(df: pd.DataFrame, label_mode: str, ehull_threshold: float) -> pd.DataFrame:
    df = df.copy()

    if label_mode == "label":
        if "label" not in df.columns:
            raise ValueError("CSV does not contain label column.")
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        return df

    if label_mode == "is_stable":
        if "is_stable" not in df.columns:
            raise ValueError("CSV does not contain is_stable column.")
        df = df[df["is_stable"].notna()].copy()
        df["label"] = df["is_stable"].astype(int)
        return df

    if label_mode == "ehull":
        if "energy_above_hull" not in df.columns:
            raise ValueError("CSV does not contain energy_above_hull column.")
        df = df[df["energy_above_hull"].notna()].copy()
        df["label"] = (df["energy_above_hull"].astype(float) <= ehull_threshold).astype(int)
        return df

    raise ValueError("label_mode must be one of: label / is_stable / ehull")


# =========================
# Feature engineering helpers
# =========================
def _infer_site_lists(row: pd.Series) -> Tuple[List[str], List[float], List[str], List[float]]:
    A_elements = _safe_json_loads(row.get("A_elements", None)) or []
    A_amounts = _safe_json_loads(row.get("A_amounts", None)) or []
    B_elements = _safe_json_loads(row.get("B_elements", None)) or []
    B_amounts = _safe_json_loads(row.get("B_amounts", None)) or []

    A_elements = [str(x) for x in A_elements]
    B_elements = [str(x) for x in B_elements]
    A_amounts = [float(x) for x in A_amounts]
    B_amounts = [float(x) for x in B_amounts]
    return A_elements, A_amounts, B_elements, B_amounts


def _dominant_and_sub(elements: List[str], amounts: List[float]) -> Tuple[Optional[str], Optional[str], float]:
    if len(elements) == 0 or len(elements) != len(amounts):
        return None, None, np.nan
    idx = int(np.argmax(amounts))
    host = elements[idx]
    if len(elements) == 1:
        return host, None, 0.0
    others = [(e, a) for i, (e, a) in enumerate(zip(elements, amounts)) if i != idx]
    if len(others) == 0:
        return host, None, 0.0
    sub, sub_amt = sorted(others, key=lambda t: t[1], reverse=True)[0]
    return host, sub, float(sub_amt)


def _row_value(row: pd.Series, key: str) -> Any:
    if key in row.index and pd.notna(row.get(key, np.nan)):
        val = row.get(key)
        if str(val).strip().lower() not in {"", "nan", "none"}:
            return val
    return None


def _first_available(row: pd.Series, keys: List[str]) -> Optional[str]:
    for key in keys:
        val = _row_value(row, key)
        if val is not None:
            return str(val)
    return None


def derive_site_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive or complete site-level metadata.

    Priority is given to existing upstream columns such as A_host/A_sub/B_host/
    B_sub and mix_ratio. Only missing values are inferred from A_elements/
    A_amounts/B_elements/B_amounts. This avoids generating chemically
    meaningless family keys such as A:None->None when the candidate pool already
    contains explicit host/substitution columns.
    """
    df = df.copy()

    if "normalized_composition_key" not in df.columns and "normalized_composition" in df.columns:
        df["normalized_composition_key"] = df["normalized_composition"].apply(canonicalize_norm_comp)

    if "mix_site" not in df.columns:
        df["mix_site"] = np.nan

    A_hosts, A_subs, B_hosts, B_subs = [], [], [], []
    mix_ratios, n_A_species, n_B_species = [], [], []
    host_displays, family_keys = [], []

    for _, row in df.iterrows():
        A_elements, A_amounts, B_elements, B_amounts = _infer_site_lists(row)
        A_host_inf, A_sub_inf, A_mix_inf = _dominant_and_sub(A_elements, A_amounts)
        B_host_inf, B_sub_inf, B_mix_inf = _dominant_and_sub(B_elements, B_amounts)

        A_host = _first_available(row, ["A_host", "A_major", "A_parent", "A"]) or A_host_inf
        A_sub = _first_available(row, ["A_sub", "A_dopant", "A_minor"]) or A_sub_inf
        B_host = _first_available(row, ["B_host", "B_major", "B_parent", "B"]) or B_host_inf
        B_sub = _first_available(row, ["B_sub", "B_dopant", "B_minor"]) or B_sub_inf

        mix_site_raw = _first_available(row, ["mix_site"])
        mix_site = str(mix_site_raw).strip() if mix_site_raw is not None else ""
        if mix_site not in {"A", "B"}:
            if A_sub is not None and B_sub is None:
                mix_site = "A"
            elif B_sub is not None and A_sub is None:
                mix_site = "B"

        mix_ratio = np.nan
        explicit_ratio = _row_value(row, "mix_ratio")
        if explicit_ratio is not None:
            try:
                mix_ratio = float(explicit_ratio)
            except Exception:
                mix_ratio = np.nan
        if np.isnan(mix_ratio):
            if mix_site == "A":
                mix_ratio = A_mix_inf
            elif mix_site == "B":
                mix_ratio = B_mix_inf
            else:
                mix_ratio = A_mix_inf if (not np.isnan(A_mix_inf) and A_mix_inf > 0) else B_mix_inf

        n_A = len(A_elements) if len(A_elements) > 0 else int(1 + (A_sub is not None))
        n_B = len(B_elements) if len(B_elements) > 0 else int(1 + (B_sub is not None))

        host_display = _first_available(
            row,
            [
                "parent_formula_pretty",
                "host_formula_pretty",
                "host_formula",
                "parent_formula",
                "parent",
                "parent_host",
                "host_id",
                "chemsys_query",
            ],
        )
        if host_display is None:
            if A_host is not None and B_host is not None:
                host_display = f"{A_host}{B_host}O3"
            else:
                host_display = "unknown_host"

        if mix_site == "A":
            family_key = f"{host_display}|A:{A_host}->{A_sub}|B:{B_host}"
        elif mix_site == "B":
            family_key = f"{host_display}|A:{A_host}|B:{B_host}->{B_sub}"
        else:
            family_key = f"{host_display}|A:{A_host}->{A_sub}|B:{B_host}->{B_sub}"

        A_hosts.append(A_host)
        A_subs.append(A_sub)
        B_hosts.append(B_host)
        B_subs.append(B_sub)
        mix_ratios.append(mix_ratio)
        n_A_species.append(n_A)
        n_B_species.append(n_B)
        host_displays.append(host_display)
        family_keys.append(family_key)

    for col, vals in [
        ("A_host", A_hosts),
        ("A_sub", A_subs),
        ("B_host", B_hosts),
        ("B_sub", B_subs),
        ("mix_ratio", mix_ratios),
        ("n_A_species", n_A_species),
        ("n_B_species", n_B_species),
        ("host_display", host_displays),
        ("family_key", family_keys),
    ]:
        if col not in df.columns:
            df[col] = vals
        else:
            df[col] = df[col].where(df[col].notna(), pd.Series(vals, index=df.index))

    # Always refresh these derived columns after filling site metadata because
    # they define family-level aggregation and should not retain stale values.
    df["host_display"] = host_displays
    df["family_key"] = family_keys

    return df

def extract_fraction_dict(norm_comp: Any) -> Dict[str, float]:
    d = _safe_json_loads(norm_comp)
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                pass
        return out
    return {}


def _composition_source_for_fraction(row: pd.Series) -> Any:
    for col in ["normalized_composition", "composition_dict"]:
        if col in row.index and pd.notna(row.get(col, np.nan)):
            return row.get(col)
    return None


def build_fraction_features(labeled_df: pd.DataFrame, pool_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Build elemental fraction features from normalized_composition when present,
    falling back to composition_dict for generated candidate pools.
    """
    element_set = set()
    for df in [labeled_df, pool_df]:
        for _, row in df.iterrows():
            comp = extract_fraction_dict(_composition_source_for_fraction(row))
            element_set.update([el for el in comp.keys() if el != "O"])

    element_list = sorted(element_set)
    frac_cols = [f"frac_{el}" for el in element_list]

    def add_frac(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rows = []
        for _, row0 in df.iterrows():
            comp = extract_fraction_dict(_composition_source_for_fraction(row0))
            row = {f"frac_{el}": float(comp.get(el, 0.0)) for el in element_list}
            rows.append(row)
        frac_df = pd.DataFrame(rows, index=df.index)
        for col in frac_cols:
            df[col] = frac_df[col].astype(float)
        return df

    return add_frac(labeled_df), add_frac(pool_df), frac_cols


# =========================
# Model utilities
# =========================
def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )


def make_train_val_split(df: pd.DataFrame, seed: int, val_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(df))
    y = df["label"].values.astype(int)
    tr_idx, va_idx = train_test_split(idx, test_size=val_size, random_state=seed, stratify=y)
    return tr_idx, va_idx


def tune_threshold(y_val: np.ndarray, probs_val: np.ndarray, objective: str = "balanced_acc") -> Tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 91)
    best_t, best_score = 0.5, -np.inf
    for t in grid:
        pred = (probs_val >= t).astype(int)
        if objective == "balanced_acc":
            score = balanced_accuracy_score(y_val, pred)
        elif objective == "f1":
            score = f1_score(y_val, pred, zero_division=0)
        else:
            raise ValueError("objective must be balanced_acc or f1")
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, float(best_score)


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (probs >= threshold).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": compute_ece(y_true, probs),
        "val_threshold_best": float(threshold),
    }
    return out


def train_logreg_pipeline(X_df: pd.DataFrame, y: np.ndarray, cat_cols: List[str], num_cols: List[str], C: float) -> Pipeline:
    pre = build_preprocessor(cat_cols, num_cols)
    clf = LogisticRegression(
        C=C,
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=None,
    )
    pipe = Pipeline([
        ("preprocessor", pre),
        ("model", clf),
    ])
    pipe.fit(X_df[cat_cols + num_cols], y)
    return pipe


def required_descriptor_columns_for_representation(representation: str) -> List[str]:
    if representation == "PiDF + ML":
        return ["wtf", "sigma_A", "sigma_B", "delta_q"]
    if representation == "Weighted TF + ML":
        return ["wtf"]
    return []


def filter_required_descriptors(
    df: pd.DataFrame,
    required_cols: List[str],
    name: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    before = len(df)
    if len(required_cols) == 0:
        return df.copy().reset_index(drop=True), {"before": int(before), "after": int(before), "removed": 0}

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required descriptor columns for scoring: {missing}")

    out = df[df[required_cols].notna().all(axis=1)].copy().reset_index(drop=True)
    return out, {"before": int(before), "after": int(len(out)), "removed": int(before - len(out))}


def get_representation_feature_set(labeled_df: pd.DataFrame, pool_df: pd.DataFrame, representation: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    labeled_df = derive_site_metadata(labeled_df)
    pool_df = derive_site_metadata(pool_df)
    labeled_df, pool_df, frac_cols = build_fraction_features(labeled_df, pool_df)

    base_cat = [c for c in ["mix_site", "A_host", "A_sub", "B_host", "B_sub"] if c in labeled_df.columns and c in pool_df.columns]
    base_num_common = [
        "mix_ratio", "n_A_species", "n_B_species",
        *frac_cols,
        "r_A_eff", "r_B_eff", "q_A_eff", "q_B_eff",
    ]
    base_num_common = [c for c in base_num_common if c in labeled_df.columns and c in pool_df.columns]

    feature_sets = {
        "Conventional Base + ML": {
            "categorical": base_cat,
            "numeric": base_num_common,
        },
        "Weighted TF + ML": {
            "categorical": base_cat,
            "numeric": base_num_common + [c for c in ["wtf"] if c in labeled_df.columns and c in pool_df.columns],
        },
        "PiDF + ML": {
            "categorical": base_cat,
            "numeric": base_num_common + [c for c in ["wtf", "sigma_A", "sigma_B", "delta_q"] if c in labeled_df.columns and c in pool_df.columns],
        },
    }

    if representation not in feature_sets:
        raise ValueError(f"Unknown representation: {representation}")
    return labeled_df, pool_df, feature_sets[representation]


# =========================
# Scoring + aggregation
# =========================
def _aggregate_dedup_keep_best(df: pd.DataFrame, dedup_key_col: str, score_col: str) -> pd.DataFrame:
    if dedup_key_col not in df.columns:
        return df.copy().reset_index(drop=True)
    tmp = df.copy()
    tmp = tmp[tmp[dedup_key_col].notna()].copy()
    idx = tmp.groupby(dedup_key_col)[score_col].idxmax()
    out = tmp.loc[idx].copy().reset_index(drop=True)
    return out


def charge_bucket(delta_q: float, charge_tol: float, conservative_deltaq: float) -> str:
    try:
        dq = abs(float(delta_q))
    except Exception:
        return "unknown"
    if dq <= charge_tol:
        return "charge_balanced"
    if dq <= conservative_deltaq:
        return "low_mismatch"
    if dq <= 0.5:
        return "moderate_mismatch"
    return "high_mismatch"


def build_candidate_formula_display(row: pd.Series) -> str:
    if pd.notna(row.get("formula_pretty", np.nan)):
        return str(row["formula_pretty"])
    if pd.notna(row.get("candidate_formula", np.nan)):
        return str(row["candidate_formula"])
    return row.get("normalized_composition_key", "unknown_formula")


def compute_family_summary(df_ranked: pd.DataFrame, top_k: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df_ranked.copy()
    if "family_key" not in df.columns:
        raise ValueError("family_key column missing.")

    top_df = df.nsmallest(top_k, columns=["rank_overall"]).copy() if "rank_overall" in df.columns else df.head(top_k).copy()
    fam_counts = top_df["family_key"].value_counts()

    concentration = {
        "top_k": int(top_k),
        "n_families_in_top_k": int(fam_counts.size),
        "top1_family_share": float(fam_counts.iloc[0] / top_k) if fam_counts.size >= 1 else 0.0,
        "top3_family_share": float(fam_counts.head(3).sum() / top_k) if fam_counts.size >= 1 else 0.0,
        "top5_family_share": float(fam_counts.head(5).sum() / top_k) if fam_counts.size >= 1 else 0.0,
        "top_family_counts": fam_counts.head(10).to_dict(),
    }

    agg = df.groupby("family_key").agg(
        host_display=("host_display", "first"),
        mix_site=("mix_site", "first"),
        A_host=("A_host", "first"),
        A_sub=("A_sub", "first"),
        B_host=("B_host", "first"),
        B_sub=("B_sub", "first"),
        n_candidates=("family_key", "size"),
        best_formula=("candidate_display", lambda x: x.iloc[np.argmax(df.loc[x.index, "score_mean"].values)] if len(x) > 0 else None),
        max_score=("score_mean", "max"),
        mean_score=("score_mean", "mean"),
        score_std_across_candidates=("score_mean", "std"),
        best_mix_ratio=("mix_ratio", lambda x: x.iloc[np.argmax(df.loc[x.index, "score_mean"].values)] if len(x) > 0 else np.nan),
        min_delta_q=("delta_q", "min"),
        max_delta_q=("delta_q", "max"),
        n_topk=("in_topk", "sum"),
    ).reset_index()

    agg = agg.sort_values(["max_score", "mean_score", "n_topk"], ascending=[False, False, False]).reset_index(drop=True)
    return agg, concentration


# =========================
# Plotting
# =========================
def plot_score_histogram(df: pd.DataFrame, save_path: str, top_k: int = 100) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    scores = df["score_mean"].astype(float).values
    ax.hist(scores, bins=30, alpha=0.85)

    if len(df) >= top_k:
        cutoff = float(df.sort_values("score_mean", ascending=False).iloc[top_k - 1]["score_mean"])
        ax.axvline(cutoff, linestyle="--", linewidth=1.4, label=f"Top-{top_k} cutoff")
        ax.legend(frameon=False, fontsize=8)

    ax.set_xlabel("Predicted near-stability score")
    ax.set_ylabel("Candidate count")
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path.lower().endswith(".png"):
        fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_score_vs_deltaq(df: pd.DataFrame, save_path: str) -> None:
    if "delta_q" not in df.columns:
        return
    plt.figure(figsize=(6, 4.5))
    plt.scatter(df["delta_q"].values, df["score_mean"].values, s=12, alpha=0.7)
    plt.xlabel(r"$\Delta q$")
    plt.ylabel("Predicted near-stability score")
    plt.title("Score vs. nominal charge mismatch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_top_family_curves(df: pd.DataFrame, top_families: List[str], save_path: str) -> None:
    plt.figure(figsize=(7, 5))
    for fam in top_families:
        sub = df[df["family_key"] == fam].copy()
        if sub.empty or "mix_ratio" not in sub.columns:
            continue
        sub = sub.sort_values("mix_ratio")
        x = sub["mix_ratio"].astype(float).values
        y = sub["score_mean"].astype(float).values
        if len(x) == 0:
            continue
        plt.plot(x, y, marker="o", label=fam)
    plt.xlabel("Mixing ratio")
    plt.ylabel("Predicted near-stability score")
    plt.title("Score versus mixing ratio for top-ranked families")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-csv", type=str, required=True, help="Labeled benchmark CSV (e.g., exp42_perovskite_like_subset.csv)")
    parser.add_argument("--pool-csv", type=str, required=True, help="Unlabeled candidate-pool CSV")
    parser.add_argument("--outdir", type=str, default="runs/section5")
    parser.add_argument("--representation", type=str, default="PiDF + ML",
                        choices=["Conventional Base + ML", "Weighted TF + ML", "PiDF + ML"])
    parser.add_argument("--label-mode", type=str, default="ehull", choices=["label", "is_stable", "ehull"])
    parser.add_argument("--ehull-threshold", type=float, default=0.05)
    parser.add_argument("--dedup-key", type=str, default="auto")
    parser.add_argument("--exclude-overlap", action="store_true", default=True,
                        help="Remove pool entries whose dedup key overlaps the labeled set.")
    parser.add_argument("--no-exclude-overlap", dest="exclude_overlap", action="store_false")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--inner-val-size", type=float, default=0.2)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--threshold-objective", type=str, default="balanced_acc", choices=["balanced_acc", "f1"])
    parser.add_argument("--charge-balanced-tol", type=float, default=1e-8)
    parser.add_argument("--conservative-deltaq", type=float, default=0.2)
    parser.add_argument("--geometry-low", type=float, default=0.8)
    parser.add_argument("--geometry-high", type=float, default=1.1)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--top-k-family", type=int, default=100)
    parser.add_argument("--top-family-curves", type=int, default=5)
    parser.add_argument(
        "--pool-scoring-mode",
        type=str,
        default="train_split",
        choices=["train_split", "full_refit"],
        help=(
            "train_split scores the pool with the seed-specific model used for internal validation; "
            "full_refit reports validation on the split but refits on the full labeled benchmark before scoring."
        ),
    )
    parser.add_argument("--save-clean-datasets", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "plots"))

    # ---------------------
    # Load + clean labeled
    # ---------------------
    labeled = pd.read_csv(args.labeled_csv)
    labeled = build_label_column(labeled, args.label_mode, args.ehull_threshold)
    labeled = derive_site_metadata(labeled)
    labeled = ensure_normalized_composition_key(labeled)

    labeled_key = choose_dedup_key(labeled, args.dedup_key)
    before_labeled = len(labeled)
    labeled = labeled[labeled[labeled_key].notna()].copy()
    if labeled_key == "normalized_composition_key" and "energy_above_hull" in labeled.columns:
        # keep the lowest-Ehull representative for training
        idx_keep = labeled.groupby(labeled_key)["energy_above_hull"].idxmin()
        labeled = labeled.loc[idx_keep].copy()
    else:
        labeled = labeled.drop_duplicates(subset=[labeled_key]).copy()
    labeled = labeled.reset_index(drop=True)

    # ---------------------
    # Load pool
    # ---------------------
    pool = pd.read_csv(args.pool_csv)
    pool = derive_site_metadata(pool)
    pool = ensure_normalized_composition_key(pool)
    pool, self_sub_removed = filter_self_substitutions(pool)
    pool_key = choose_dedup_key(pool, args.dedup_key)

    # representation feature space
    labeled, pool, feat = get_representation_feature_set(labeled, pool, args.representation)

    # Do not impute core PiDF descriptors for ranking; missing descriptor rows are
    # removed and counted explicitly.
    required_desc = required_descriptor_columns_for_representation(args.representation)
    labeled, labeled_descriptor_filter = filter_required_descriptors(labeled, required_desc, name="labeled benchmark")
    pool, pool_descriptor_filter = filter_required_descriptors(pool, required_desc, name="candidate pool")

    cat_cols = feat["categorical"]
    num_cols = feat["numeric"]

    # Keep all non-overlapping pool occurrences for scoring. Deduplication is
    # performed after scoring so that the highest-scoring occurrence is retained.
    pool = pool[pool[pool_key].notna()].copy().reset_index(drop=True)

    # Optional overlap removal
    overlap_removed = 0
    if args.exclude_overlap:
        labeled_keys = set(labeled[labeled_key].dropna().astype(str).tolist())
        mask_overlap = pool[pool_key].astype(str).isin(labeled_keys) if pool_key in pool.columns else pd.Series(False, index=pool.index)
        overlap_removed = int(mask_overlap.sum())
        if overlap_removed > 0:
            pool = pool.loc[~mask_overlap].copy().reset_index(drop=True)

    if args.save_clean_datasets:
        labeled.to_csv(os.path.join(args.outdir, "clean_labeled_benchmark.csv"), index=False)
        pool.to_csv(os.path.join(args.outdir, "clean_candidate_pool_before_scoring.csv"), index=False)

    # ---------------------
    # Train ensemble + score
    # ---------------------
    validation_records: List[Dict[str, Any]] = []
    all_pool_scores = []

    X_cols = cat_cols + num_cols
    missing_labeled = [c for c in X_cols if c not in labeled.columns]
    missing_pool = [c for c in X_cols if c not in pool.columns]
    if missing_labeled:
        raise ValueError(f"Labeled CSV missing required feature columns: {missing_labeled}")
    if missing_pool:
        # fill missing pool numeric/categorical columns with NaN so imputer handles them
        for c in missing_pool:
            pool[c] = np.nan

    for seed in args.seeds:
        set_seed(seed)
        tr_idx, va_idx = make_train_val_split(labeled, seed=seed, val_size=args.inner_val_size)
        tr_df = labeled.iloc[tr_idx].reset_index(drop=True)
        va_df = labeled.iloc[va_idx].reset_index(drop=True)

        y_tr = tr_df["label"].values.astype(int)
        y_va = va_df["label"].values.astype(int)

        pipe = train_logreg_pipeline(tr_df, y_tr, cat_cols=cat_cols, num_cols=num_cols, C=args.logreg_c)
        probs_va = pipe.predict_proba(va_df[X_cols])[:, 1]
        best_t, best_val_score = tune_threshold(y_va, probs_va, objective=args.threshold_objective)
        metrics = compute_metrics(y_va, probs_va, threshold=best_t)
        metrics["seed"] = seed
        metrics["val_balanced_acc_best"] = best_val_score
        validation_records.append(metrics)

        if args.pool_scoring_mode == "full_refit":
            scoring_pipe = train_logreg_pipeline(
                labeled,
                labeled["label"].values.astype(int),
                cat_cols=cat_cols,
                num_cols=num_cols,
                C=args.logreg_c,
            )
        else:
            scoring_pipe = pipe

        pool_probs = scoring_pipe.predict_proba(pool[X_cols])[:, 1]
        all_pool_scores.append(pool_probs)

    score_mat = np.vstack(all_pool_scores).T  # [n_pool, n_seeds]
    pool_scored = pool.copy()
    for i, seed in enumerate(args.seeds):
        pool_scored[f"score_seed_{seed}"] = score_mat[:, i]
    pool_scored["score_mean"] = score_mat.mean(axis=1)
    pool_scored["score_std"] = score_mat.std(axis=1)
    pool_scored["candidate_display"] = pool_scored.apply(build_candidate_formula_display, axis=1)
    pool_scored["charge_bucket"] = pool_scored["delta_q"].apply(
        lambda x: charge_bucket(x, args.charge_balanced_tol, args.conservative_deltaq)
    ) if "delta_q" in pool_scored.columns else "unknown"
    if "wtf" in pool_scored.columns:
        pool_scored["geometry_conservative"] = pool_scored["wtf"].between(args.geometry_low, args.geometry_high)
    else:
        pool_scored["geometry_conservative"] = False
    if "delta_q" in pool_scored.columns:
        pool_scored["chemically_conservative"] = (
            (pool_scored["delta_q"].abs() <= args.conservative_deltaq) & pool_scored["geometry_conservative"].astype(bool)
        )
        pool_scored["charge_balanced"] = pool_scored["delta_q"].abs() <= args.charge_balanced_tol
    else:
        pool_scored["chemically_conservative"] = pool_scored["geometry_conservative"]
        pool_scored["charge_balanced"] = False

    pool_scored_raw = pool_scored.sort_values("score_mean", ascending=False).reset_index(drop=True)
    pool_scored_raw.to_csv(os.path.join(args.outdir, "scored_pool_raw.csv"), index=False)

    invalid_family_mask = pool_scored_raw["family_key"].astype(str).str.contains("None|nan", case=False, regex=True)
    n_invalid_family_keys = int(invalid_family_mask.sum())

    # Deduplicated ranking: keep the best-scoring occurrence of each composition/formula
    pool_ranked = _aggregate_dedup_keep_best(pool_scored_raw, pool_key, "score_mean")
    pool_ranked = pool_ranked.sort_values("score_mean", ascending=False).reset_index(drop=True)
    pool_ranked["rank_overall"] = np.arange(1, len(pool_ranked) + 1)
    pool_ranked["in_topk"] = pool_ranked["rank_overall"] <= int(args.top_k_family)
    pool_ranked.to_csv(os.path.join(args.outdir, "scored_pool_deduplicated.csv"), index=False)

    # Top candidate lists
    top_overall = pool_ranked.head(args.top_n).copy()
    top_overall.to_csv(os.path.join(args.outdir, "top_overall_candidates.csv"), index=False)

    top_cons = pool_ranked[pool_ranked["chemically_conservative"].astype(bool)].head(args.top_n).copy()
    top_cons.to_csv(os.path.join(args.outdir, "top_chemically_conservative_candidates.csv"), index=False)

    top_bal = pool_ranked[pool_ranked["charge_balanced"].astype(bool)].head(args.top_n).copy()
    top_bal.to_csv(os.path.join(args.outdir, "top_charge_balanced_candidates.csv"), index=False)

    # Family-level summary
    fam_summary, concentration = compute_family_summary(pool_ranked, top_k=args.top_k_family)
    fam_summary.to_csv(os.path.join(args.outdir, "family_summary.csv"), index=False)
    with open(os.path.join(args.outdir, "family_concentration.json"), "w", encoding="utf-8") as f:
        json.dump(concentration, f, ensure_ascii=False, indent=2)

    top_families = fam_summary["family_key"].head(args.top_family_curves).tolist()
    curve_df = pool_ranked[pool_ranked["family_key"].isin(top_families)].copy()
    curve_df = curve_df.sort_values(["family_key", "mix_ratio", "score_mean"], ascending=[True, True, False])
    curve_df.to_csv(os.path.join(args.outdir, "score_vs_mixratio_top_families.csv"), index=False)

    # Plots
    plot_score_histogram(pool_ranked, os.path.join(args.outdir, "plots", "score_histogram.png"), top_k=args.top_k_family)
    if "delta_q" in pool_ranked.columns:
        plot_score_vs_deltaq(pool_ranked, os.path.join(args.outdir, "plots", "score_vs_deltaq.png"))
    plot_top_family_curves(pool_ranked, top_families, os.path.join(args.outdir, "plots", "top_family_curves.png"))

    # Validation summary
    val_df = pd.DataFrame(validation_records)
    val_df.to_csv(os.path.join(args.outdir, "labeled_internal_validation.csv"), index=False)
    labeled_val_summary = {
        metric: {
            "mean": float(val_df[metric].mean()),
            "std": float(val_df[metric].std(ddof=0))
        }
        for metric in [
            "roc_auc", "pr_auc", "f1", "balanced_acc", "precision", "recall", "brier", "ece", "val_threshold_best"
        ]
    }
    with open(os.path.join(args.outdir, "labeled_internal_validation.json"), "w", encoding="utf-8") as f:
        json.dump(labeled_val_summary, f, ensure_ascii=False, indent=2)

    # Summary JSON
    summary = {
        "metadata": {
            "representation": args.representation,
            "label_mode": args.label_mode,
            "ehull_threshold": args.ehull_threshold,
            "seeds": args.seeds,
            "inner_val_size": args.inner_val_size,
            "threshold_objective": args.threshold_objective,
            "logreg_c": args.logreg_c,
            "dedup_key_labeled": labeled_key,
            "dedup_key_pool": pool_key,
            "pool_self_substitutions_removed": int(self_sub_removed),
            "exclude_overlap": args.exclude_overlap,
            "overlap_removed_from_pool": int(overlap_removed),
            "descriptor_filter_labeled": labeled_descriptor_filter,
            "descriptor_filter_pool": pool_descriptor_filter,
            "pool_scoring_mode": args.pool_scoring_mode,
            "n_labeled_before_dedup": int(before_labeled),
            "n_labeled_after_dedup": int(len(labeled)),
            "n_pool_before_scoring": int(len(pool)),
            "n_pool_raw": int(len(pool_scored_raw)),
            "n_pool_dedup": int(len(pool_ranked)),
            "n_invalid_family_keys_before_dedup": int(n_invalid_family_keys),
            "charge_balanced_tol": args.charge_balanced_tol,
            "conservative_deltaq": args.conservative_deltaq,
            "geometry_window": [args.geometry_low, args.geometry_high],
            "top_n": int(args.top_n),
            "top_k_family": int(args.top_k_family),
        },
        "feature_set": feat,
        "labeled_internal_validation": labeled_val_summary,
        "top_overall_head": top_overall[[c for c in [
            "rank_overall", "candidate_display", "host_display", "mix_site", "mix_ratio",
            "score_mean", "score_std", "wtf", "sigma_A", "sigma_B", "delta_q", "chemically_conservative"
        ] if c in top_overall.columns]].to_dict(orient="records"),
        "top_conservative_head": top_cons[[c for c in [
            "rank_overall", "candidate_display", "host_display", "mix_site", "mix_ratio",
            "score_mean", "score_std", "wtf", "sigma_A", "sigma_B", "delta_q"
        ] if c in top_cons.columns]].to_dict(orient="records"),
        "family_concentration": concentration,
        "sanity_checks": {
            "n_topN_overall": int(len(top_overall)),
            "n_topN_chemically_conservative_within_overall": int(top_overall["chemically_conservative"].astype(bool).sum()) if "chemically_conservative" in top_overall.columns else 0,
            "n_topN_charge_balanced_within_overall": int(top_overall["charge_balanced"].astype(bool).sum()) if "charge_balanced" in top_overall.columns else 0,
            "n_topN_high_mismatch_within_overall": int((top_overall["charge_bucket"] == "high_mismatch").sum()) if "charge_bucket" in top_overall.columns else 0,
            "n_topN_chemically_conservative_list": int(len(top_cons)),
            "n_topN_charge_balanced_list": int(len(top_bal)),
            "mean_delta_q_topN": float(top_overall["delta_q"].mean()) if "delta_q" in top_overall.columns and len(top_overall) > 0 else np.nan,
            "mean_delta_q_topN_conservative_list": float(top_cons["delta_q"].mean()) if "delta_q" in top_cons.columns and len(top_cons) > 0 else np.nan,
        }
    }
    with open(os.path.join(args.outdir, "section5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("===== Section 5 candidate prioritization completed =====")
    print(json.dumps(summary["metadata"], ensure_ascii=False, indent=2))
    print(f"Results directory: {args.outdir}")


if __name__ == "__main__":
    main()
