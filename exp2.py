#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 2 script for:
Robustness under Chemically Disjoint Splits

This script is built on top of the user's original benchmark code and the
Section-1 benchmark logic, but focuses on stricter chemically disjoint
evaluation. It adds:

1) Chemically disjoint split protocols
   - host-disjoint split (via host_id / chemsys_query / parent_material_id)
   - substitution-family-disjoint split (all mixing ratios in one family held together)

2) Optional composition-level deduplication before splitting
   - canonicalized normalized_composition is preferred
   - falls back to formula_pretty if needed

3) The same baseline family as Section 1, but now under group-disjoint splits
   - Classical TF Threshold
   - Composition Vector + ML
   - Conventional Composition/Site + ML
   - Raw Radius/Valence + ML
   - Classical TF + ML
   - Weighted TF + ML
   - PiDF + ML

4) Group-level leakage diagnostics
   - number of unique groups in train / val / test
   - explicit overlap checks between splits

Recommended use for Section 2 of Results and Discussion:
- keep the same learner as Section 1 (e.g. logreg)
- enable deduplication
- run both host and family protocols
- use the compact method set for the main text, and the full set in SI
"""

import os
import ast
import json
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =========================
# Basic utilities
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            d = json.loads(s)
            if isinstance(d, dict):
                return {str(k): float(v) for k, v in d.items()}
        except Exception:
            pass
        try:
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                return {str(k): float(v) for k, v in d.items()}
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


def build_label_column(df: pd.DataFrame, label_mode: str, ehull_threshold: float) -> pd.DataFrame:
    df = df.copy()

    if label_mode == "label":
        if "label" not in df.columns:
            raise ValueError("CSV 中没有 label 列")
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        return df

    if label_mode == "is_stable":
        if "is_stable" not in df.columns:
            raise ValueError("CSV 中没有 is_stable 列")
        df = df[df["is_stable"].notna()].copy()
        df["label"] = df["is_stable"].astype(int)
        return df

    if label_mode == "ehull":
        if "energy_above_hull" not in df.columns:
            raise ValueError("CSV 中没有 energy_above_hull 列")
        df = df[df["energy_above_hull"].notna()].copy()
        df["label"] = (df["energy_above_hull"].astype(float) <= ehull_threshold).astype(int)
        return df

    raise ValueError("label_mode 只能是 label / is_stable / ehull")


# =========================
# Feature engineering
# =========================
def enrich_composition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive host/substitution and mixing-ratio columns from A/B element+amount lists.
    """
    out = df.copy()
    needed = {"A_elements", "A_amounts", "B_elements", "B_amounts"}
    if not needed.issubset(out.columns):
        return out

    records = []
    for _, row in out.iterrows():
        mix_site = row.get("mix_site", None)
        A_elements = [str(v) for v in _parse_list_like(row.get("A_elements"))]
        A_amounts = [_safe_float(v) for v in _parse_list_like(row.get("A_amounts"))]
        B_elements = [str(v) for v in _parse_list_like(row.get("B_elements"))]
        B_amounts = [_safe_float(v) for v in _parse_list_like(row.get("B_amounts"))]

        rec: Dict[str, Any] = {
            "A_host": None,
            "A_sub": None,
            "B_host": None,
            "B_sub": None,
            "mix_ratio": row.get("mix_ratio", None),
            "n_A_species": len(A_elements),
            "n_B_species": len(B_elements),
        }

        def choose_host_sub(elements: List[str], amounts: List[Optional[float]]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
            pairs: List[Tuple[str, float]] = []
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
        else:
            if len(A_elements) >= 1:
                A_sorted = sorted(A_elements)
                rec["A_host"] = A_sorted[0]
                rec["A_sub"] = A_sorted[1] if len(A_sorted) > 1 else None
            if len(B_elements) >= 1:
                B_sorted = sorted(B_elements)
                rec["B_host"] = B_sorted[0]
                rec["B_sub"] = B_sorted[1] if len(B_sorted) > 1 else None

        records.append(rec)

    derived = pd.DataFrame(records, index=out.index)
    for col in derived.columns:
        if col not in out.columns:
            out[col] = derived[col]
        else:
            out[col] = out[col].where(out[col].notna(), derived[col])
    return out


def enrich_fraction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build element-fraction features from normalized_composition. Oxygen is omitted.
    """
    out = df.copy()
    if "normalized_composition" not in out.columns:
        return out

    parsed = []
    elements = set()
    for x in out["normalized_composition"].tolist():
        d = _parse_dict_like(x)
        parsed.append(d)
        for k in d.keys():
            if str(k) != "O":
                elements.add(str(k))

    if len(elements) == 0:
        return out

    elements = sorted(elements)
    frac_data = {}
    for el in elements:
        frac_data[f"frac_{el}"] = [float(d.get(el, 0.0)) for d in parsed]

    frac_df = pd.DataFrame(frac_data, index=out.index)
    for col in frac_df.columns:
        out[col] = frac_df[col]
    return out


def canonicalize_normalized_composition(df: pd.DataFrame, ndigits: int = 8) -> pd.DataFrame:
    """
    Build a stable composition key for composition-level deduplication.

    The normalized_composition column may contain floating-point values such as
    0.30000000000000004 and 0.3 for the same nominal composition. Rounding the
    amounts before serialization makes the deduplication more robust.
    """
    out = df.copy()
    if "normalized_composition" not in out.columns:
        return out

    canonical = []
    for x in out["normalized_composition"].tolist():
        d = _parse_dict_like(x)
        items = sorted(
            [(str(k), round(float(v), ndigits)) for k, v in d.items()],
            key=lambda kv: kv[0],
        )
        canonical.append(json.dumps(items, ensure_ascii=False))
    out["normalized_composition_canonical"] = canonical
    return out


def build_family_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group all compositions belonging to the same substitution family together,
    regardless of mixing ratio and regardless of which species is the majority.

    Important:
    The family key is built from the sorted mixed-site element pair, not from
    host/sub labels inferred from the majority species. This ensures that, e.g.,
    Nd0.6Ce0.4ScO3 and Nd0.4Ce0.6ScO3 are assigned to the same A-site family.
    """
    out = df.copy()
    family_keys = []

    for _, row in out.iterrows():
        mix_site = str(row.get("mix_site", "NA"))

        A_elements = sorted([str(v) for v in _parse_list_like(row.get("A_elements"))])
        B_elements = sorted([str(v) for v in _parse_list_like(row.get("B_elements"))])

        # Fallback to enriched host/sub columns if the original element lists
        # are unavailable for any reason.
        if len(A_elements) == 0:
            A_elements = sorted(
                [
                    str(v)
                    for v in [row.get("A_host", None), row.get("A_sub", None)]
                    if v is not None and str(v) not in {"", "nan", "None"}
                ]
            )
        if len(B_elements) == 0:
            B_elements = sorted(
                [
                    str(v)
                    for v in [row.get("B_host", None), row.get("B_sub", None)]
                    if v is not None and str(v) not in {"", "nan", "None"}
                ]
            )

        A_key = "-".join(A_elements) if len(A_elements) > 0 else "NA"
        B_key = "-".join(B_elements) if len(B_elements) > 0 else "NA"

        if mix_site == "A":
            # A-site mixed: same A pair + same B species = one family.
            family = f"A|Apair={A_key}|B={B_key}"
        elif mix_site == "B":
            # B-site mixed: same B pair + same A species = one family.
            family = f"B|A={A_key}|Bpair={B_key}"
        else:
            family = f"NA|A={A_key}|B={B_key}"

        family_keys.append(family)

    out["family_group"] = family_keys
    return out


def choose_host_group_column(df: pd.DataFrame) -> str:
    for col in ["host_id", "chemsys_query", "parent_material_id"]:
        if col in df.columns:
            return col
    raise ValueError("host-disjoint split 需要 host_id / chemsys_query / parent_material_id 之一。")


def resolve_group_series(df: pd.DataFrame, protocol: str) -> Tuple[pd.Series, str]:
    if protocol == "host":
        col = choose_host_group_column(df)
        return df[col].astype(str), col
    if protocol == "family":
        if "family_group" not in df.columns:
            raise ValueError("family_group 列不存在，请先调用 build_family_group。")
        return df["family_group"].astype(str), "family_group"
    raise ValueError("protocol 只能是 host / family")


def deduplicate_dataset(df: pd.DataFrame, dedup_key: str) -> pd.DataFrame:
    out = df.copy()
    if dedup_key == "none":
        return out

    if dedup_key == "auto":
        if "normalized_composition_canonical" in out.columns:
            dedup_key = "normalized_composition_canonical"
        elif "formula_pretty" in out.columns:
            dedup_key = "formula_pretty"
        else:
            return out

    if dedup_key not in out.columns:
        raise ValueError(f"dedup_key={dedup_key} 不在数据中。")

    if "energy_above_hull" in out.columns:
        out = out.sort_values(by="energy_above_hull", ascending=True, kind="mergesort")
    out = out.drop_duplicates(subset=[dedup_key], keep="first").reset_index(drop=True)
    return out


# =========================
# Split utilities
# =========================
def _all_subsets_have_both_classes(y: np.ndarray, subsets: List[np.ndarray]) -> bool:
    for idx in subsets:
        vals = np.unique(y[idx])
        if len(vals) < 2:
            return False
    return True


def make_group_split(
    df: pd.DataFrame,
    groups: np.ndarray,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    max_trials: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(df))
    y = df["label"].values
    val_ratio = val_size / (1.0 - test_size)

    for trial in range(max_trials):
        rseed = seed + 1000 * trial
        splitter1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rseed)
        trainval_idx, test_idx = next(splitter1.split(indices, y, groups=groups))

        groups_trainval = groups[trainval_idx]
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=rseed)
        rel_train_idx, rel_val_idx = next(
            splitter2.split(trainval_idx, y[trainval_idx], groups=groups_trainval)
        )

        train_idx = trainval_idx[rel_train_idx]
        val_idx = trainval_idx[rel_val_idx]

        if _all_subsets_have_both_classes(y, [train_idx, val_idx, test_idx]):
            return train_idx, val_idx, test_idx

    raise RuntimeError(
        f"group split 在 {max_trials} 次尝试后仍未获得 train/val/test 都含有两类样本的划分。"
    )


def split_diagnostics(
    groups: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, int]:
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    test_groups = set(groups[test_idx].tolist())

    return {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_train_pos": int(y[train_idx].sum()),
        "n_val_pos": int(y[val_idx].sum()),
        "n_test_pos": int(y[test_idx].sum()),
        "n_train_neg": int(len(train_idx) - y[train_idx].sum()),
        "n_val_neg": int(len(val_idx) - y[val_idx].sum()),
        "n_test_neg": int(len(test_idx) - y[test_idx].sum()),
        "n_train_groups": int(len(train_groups)),
        "n_val_groups": int(len(val_groups)),
        "n_test_groups": int(len(test_groups)),
        "train_val_overlap": int(len(train_groups.intersection(val_groups))),
        "train_test_overlap": int(len(train_groups.intersection(test_groups))),
        "val_test_overlap": int(len(val_groups.intersection(test_groups))),
    }


# =========================
# Threshold baseline
# =========================
def threshold_predict(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return ((x >= low) & (x <= high)).astype(int)


def tune_tf_threshold(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[float, float, float]:
    x_train = train_df["classical_t"].values.astype(float)
    x_val = val_df["classical_t"].values.astype(float)
    y_val = val_df["label"].values.astype(int)

    low_grid = np.quantile(x_train, np.linspace(0.05, 0.50, 20))
    high_grid = np.quantile(x_train, np.linspace(0.50, 0.95, 20))

    best_score = -np.inf
    best_low, best_high = float(np.min(x_train)), float(np.max(x_train))

    for low in low_grid:
        for high in high_grid:
            if low >= high:
                continue
            pred = threshold_predict(x_val, low, high)
            score = balanced_accuracy_score(y_val, pred)
            if score > best_score:
                best_score = score
                best_low, best_high = float(low), float(high)

    return best_low, best_high, best_score


# =========================
# Models
# =========================
class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@dataclass
class TrainConfig:
    model_type: str = "logreg"  # mlp / logreg / rf
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    batch_size: int = 64
    patience: int = 30
    threshold_objective: str = "balanced_acc"  # balanced_acc / f1
    device: str = "cpu"
    logreg_c: float = 1.0
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = None


def build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_prob_mlp(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    x = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(x)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def train_mlp(X_train, y_train, X_val, y_val, cfg: TrainConfig) -> nn.Module:
    train_loader = build_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    model = SimpleMLP(in_dim=X_train.shape[1]).to(cfg.device)

    pos_num = max(float(y_train.sum()), 1.0)
    neg_num = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg_num / pos_num], dtype=torch.float32, device=cfg.device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_auc = -np.inf
    best_state = None
    wait = 0

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_probs = predict_prob_mlp(model, X_val, cfg.device)
        try:
            val_auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            val_auc = -np.inf

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def fit_predict_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg: TrainConfig,
):
    if cfg.model_type == "mlp":
        model = train_mlp(X_train, y_train, X_val, y_val, cfg)
        val_probs = predict_prob_mlp(model, X_val, cfg.device)
        test_probs = predict_prob_mlp(model, X_test, cfg.device)
        return model, val_probs, test_probs

    if cfg.model_type == "logreg":
        model = LogisticRegression(
            C=cfg.logreg_c,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            n_jobs=None,
        )
        model.fit(X_train, y_train.astype(int))
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        return model, val_probs, test_probs

    if cfg.model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train.astype(int))
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        return model, val_probs, test_probs

    raise ValueError("model_type 只能是 mlp / logreg / rf")


# =========================
# Metrics
# =========================
def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Binary expected calibration error.

    For each probability bin, compare the empirical positive rate with the
    mean predicted probability. This is better aligned with probability
    calibration than comparing hard-threshold accuracy with confidence.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)

        if not np.any(mask):
            continue

        bin_conf = np.mean(probs[mask])
        bin_pos_rate = np.mean(y_true[mask])
        ece += (np.sum(mask) / n) * abs(bin_pos_rate - bin_conf)

    return float(ece)


def tune_probability_threshold(y_val: np.ndarray, val_probs: np.ndarray, objective: str) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thr = 0.5
    best_score = -np.inf
    for thr in thresholds:
        pred = (val_probs >= thr).astype(int)
        if objective == "f1":
            score = f1_score(y_val, pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y_val, pred)
        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)
    return best_thr, best_score


def compute_metrics(
    y_true: np.ndarray,
    probs: Optional[np.ndarray] = None,
    pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    compute_auc: bool = True,
    include_calibration: bool = True,
) -> Dict[str, float]:
    if pred is None:
        if probs is None:
            raise ValueError("compute_metrics 需要 probs 或 pred 之一")
        pred = (probs >= threshold).astype(int)

    out: Dict[str, float] = {}
    if compute_auc and probs is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, probs))
        except ValueError:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, probs))
        except ValueError:
            out["pr_auc"] = float("nan")

    out["f1"] = float(f1_score(y_true, pred, zero_division=0))
    out["balanced_acc"] = float(balanced_accuracy_score(y_true, pred))
    out["precision"] = float(precision_score(y_true, pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, pred, zero_division=0))

    if include_calibration and probs is not None:
        try:
            out["brier"] = float(brier_score_loss(y_true, probs))
        except ValueError:
            out["brier"] = float("nan")
        out["ece"] = compute_ece(y_true, probs)

    return out


# =========================
# Feature sets
# =========================
def get_feature_sets(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    frac_num = sorted([c for c in df.columns if c.startswith("frac_")])

    conventional_cat = [c for c in ["mix_site", "A_host", "A_sub", "B_host", "B_sub"] if c in df.columns]
    identity_num = [c for c in ["mix_ratio", "n_A_species", "n_B_species"] if c in df.columns]
    raw_cat = [c for c in ["mix_site"] if c in df.columns]
    raw_num = [c for c in ["mix_ratio", "r_A_eff", "r_B_eff", "q_A_eff", "q_B_eff", "r_O"] if c in df.columns]
    conventional_num = list(dict.fromkeys(identity_num + frac_num + raw_num))

    feature_sets = {
        "Composition Vector + ML": {
            "categorical": [c for c in ["mix_site"] if c in df.columns],
            "numeric": list(dict.fromkeys(identity_num + frac_num)),
        },
        "Conventional Composition/Site + ML": {
            "categorical": conventional_cat,
            "numeric": conventional_num,
        },
        "Raw Radius/Valence + ML": {
            "categorical": raw_cat,
            "numeric": raw_num,
        },
        "Classical TF + ML": {
            "categorical": conventional_cat,
            "numeric": list(dict.fromkeys(conventional_num + [c for c in ["classical_t"] if c in df.columns])),
        },
        "Weighted TF + ML": {
            "categorical": conventional_cat,
            "numeric": list(dict.fromkeys(conventional_num + [c for c in ["wtf"] if c in df.columns])),
        },
        "PiDF + ML": {
            "categorical": conventional_cat,
            "numeric": list(dict.fromkeys(conventional_num + [c for c in ["wtf", "sigma_A", "sigma_B", "delta_q"] if c in df.columns])),
        },
    }

    for name, spec in feature_sets.items():
        if len(spec["categorical"]) + len(spec["numeric"]) == 0:
            raise ValueError(f"特征集 {name} 为空，请检查数据列。")
    return feature_sets


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


# =========================
# Single-run experiments
# =========================
def run_threshold_baseline_grouped(
    df: pd.DataFrame,
    groups: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float,
) -> Dict[str, Any]:
    train_idx, val_idx, test_idx = make_group_split(df, groups, seed, test_size=test_size, val_size=val_size)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    low, high, val_score = tune_tf_threshold(train_df, val_df)
    x_test = test_df["classical_t"].values.astype(float)
    y_test = test_df["label"].values.astype(int)
    pred = threshold_predict(x_test, low, high)

    metrics = compute_metrics(y_true=y_test, pred=pred, compute_auc=False, include_calibration=False)
    metrics["val_balanced_acc_best"] = float(val_score)

    return {
        "metrics": metrics,
        "threshold_low": float(low),
        "threshold_high": float(high),
        "split_info": split_diagnostics(groups, df["label"].values.astype(int), train_idx, val_idx, test_idx),
    }


def run_ml_baseline_grouped(
    df: pd.DataFrame,
    groups: np.ndarray,
    seed: int,
    cat_cols: List[str],
    num_cols: List[str],
    cfg: TrainConfig,
    test_size: float,
    val_size: float,
) -> Dict[str, Any]:
    train_idx, val_idx, test_idx = make_group_split(df, groups, seed, test_size=test_size, val_size=val_size)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train = preprocessor.fit_transform(train_df[cat_cols + num_cols])
    X_val = preprocessor.transform(val_df[cat_cols + num_cols])
    X_test = preprocessor.transform(test_df[cat_cols + num_cols])

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()

    y_train = train_df["label"].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    _, val_probs, test_probs = fit_predict_model(X_train, y_train, X_val, y_val, X_test, cfg)
    best_thr, best_val_score = tune_probability_threshold(y_val.astype(int), val_probs, cfg.threshold_objective)
    metrics = compute_metrics(y_true=y_test.astype(int), probs=test_probs, threshold=best_thr, compute_auc=True)
    metrics["val_threshold_best"] = float(best_thr)
    metrics[f"val_{cfg.threshold_objective}_best"] = float(best_val_score)
    metrics["n_features"] = int(X_train.shape[1])

    return {
        "metrics": metrics,
        "y_test": y_test.astype(int),
        "probs": test_probs,
        "split_info": split_diagnostics(groups, df["label"].values.astype(int), train_idx, val_idx, test_idx),
    }


# =========================
# Summaries
# =========================
def summarize(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metric_names = list(results[0]["metrics"].keys())
    out = {}
    for metric in metric_names:
        vals = np.array([r["metrics"][metric] for r in results], dtype=float)
        out[metric] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
        }
    return out


def summarize_split_info(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    keys = list(results[0]["split_info"].keys())
    out = {}
    for key in keys:
        vals = np.array([r["split_info"][key] for r in results], dtype=float)
        out[key] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
        }
    return out


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./mp_data/exp42_perovskite_like_subset.csv")
    parser.add_argument("--outdir", type=str, default="runs/section2_group_robustness")
    parser.add_argument("--protocols", type=str, nargs="+", default=["host", "family"], choices=["host", "family"])
    parser.add_argument("--method-set", type=str, default="compact", choices=["compact", "full"])
    parser.add_argument("--dedup-key", type=str, default="auto", choices=["auto", "none", "normalized_composition_canonical", "formula_pretty"])
    parser.add_argument("--label-mode", type=str, default="ehull", choices=["label", "is_stable", "ehull"])
    parser.add_argument("--ehull-threshold", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--model-type", type=str, default="logreg", choices=["mlp", "logreg", "rf"])
    parser.add_argument("--threshold-objective", type=str, default="balanced_acc", choices=["balanced_acc", "f1"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    set_seed(0)

    # Load and preprocess
    df = pd.read_csv(args.csv)
    before_label = len(df)
    df = build_label_column(df, args.label_mode, args.ehull_threshold)
    after_label = len(df)

    required_descriptor_cols = ["classical_t", "wtf", "sigma_A", "sigma_B", "delta_q"]
    for col in required_descriptor_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    df = df[df[required_descriptor_cols].notna().all(axis=1)].copy()

    df = enrich_composition_columns(df)
    df = enrich_fraction_features(df)
    df = canonicalize_normalized_composition(df)
    df = build_family_group(df)

    duplicate_stats_before = {
        "n_duplicate_normalized_composition": int(
            len(df) - df["normalized_composition_canonical"].nunique()
        ) if "normalized_composition_canonical" in df.columns else None,
        "n_duplicate_formula_pretty": int(
            len(df) - df["formula_pretty"].astype(str).nunique()
        ) if "formula_pretty" in df.columns else None,
    }

    before_dedup = len(df)
    df = deduplicate_dataset(df, args.dedup_key)
    after_dedup = len(df)

    if len(df) == 0:
        raise ValueError("清洗后没有可用样本。")

    # Rebuild family groups after deduplication to keep the group labels
    # consistent with the final evaluated dataset.
    df = build_family_group(df)

    group_count_after_dedup = {
        "n_host_groups": int(resolve_group_series(df, "host")[0].nunique()) if len(df) > 0 else 0,
        "n_family_groups": int(resolve_group_series(df, "family")[0].nunique()) if len(df) > 0 else 0,
    }

    feature_sets = get_feature_sets(df)

    cfg = TrainConfig(
        model_type=args.model_type,
        threshold_objective=args.threshold_objective,
        device=args.device,
        logreg_c=args.logreg_c,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
    )

    if args.method_set == "compact":
        method_names = [
            "Classical TF Threshold",
            "Conventional Composition/Site + ML",
            "Weighted TF + ML",
            "PiDF + ML",
        ]
    else:
        method_names = [
            "Classical TF Threshold",
            "Composition Vector + ML",
            "Conventional Composition/Site + ML",
            "Raw Radius/Valence + ML",
            "Classical TF + ML",
            "Weighted TF + ML",
            "PiDF + ML",
        ]

    summary = {
        "metadata": {
            "n_samples_raw": int(before_label),
            "n_samples_after_label": int(after_label),
            "n_samples_after_dedup": int(len(df)),
            "n_stable": int((df["label"] == 1).sum()),
            "n_unstable": int((df["label"] == 0).sum()),
            "positive_rate": float(df["label"].mean()),
            "label_mode": args.label_mode,
            "ehull_threshold": args.ehull_threshold,
            "protocols": args.protocols,
            "method_set": args.method_set,
            "dedup_key": args.dedup_key,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "seeds": args.seeds,
            "train_config": asdict(cfg),
            "duplicate_stats_before_dedup": duplicate_stats_before,
            "dedup_counts": {
                "before": int(before_dedup),
                "after": int(after_dedup),
            },
            "group_count_after_dedup": group_count_after_dedup,
        },
        "feature_sets": feature_sets,
        "protocol_results": {},
    }

    per_run_rows = []
    diagnostics_rows = []
    main_rows = []
    calibration_rows = []

    for protocol in args.protocols:
        groups, group_source = resolve_group_series(df, protocol)
        protocol_results = {name: [] for name in method_names}

        for seed in args.seeds:
            set_seed(seed)

            if "Classical TF Threshold" in method_names:
                res = run_threshold_baseline_grouped(
                    df=df,
                    groups=groups.values,
                    seed=seed,
                    test_size=args.test_size,
                    val_size=args.val_size,
                )
                protocol_results["Classical TF Threshold"].append(res)
                row = {"protocol": protocol, "group_source": group_source, "seed": seed, "method": "Classical TF Threshold"}
                row.update(res["metrics"])
                per_run_rows.append(row)
                diag = {"protocol": protocol, "group_source": group_source, "seed": seed, "method": "Classical TF Threshold"}
                diag.update(res["split_info"])
                diagnostics_rows.append(diag)

            for name in [m for m in method_names if m != "Classical TF Threshold"]:
                spec = feature_sets[name]
                res = run_ml_baseline_grouped(
                    df=df,
                    groups=groups.values,
                    seed=seed,
                    cat_cols=spec["categorical"],
                    num_cols=spec["numeric"],
                    cfg=cfg,
                    test_size=args.test_size,
                    val_size=args.val_size,
                )
                protocol_results[name].append(res)
                row = {"protocol": protocol, "group_source": group_source, "seed": seed, "method": name}
                row.update(res["metrics"])
                per_run_rows.append(row)
                diag = {"protocol": protocol, "group_source": group_source, "seed": seed, "method": name}
                diag.update(res["split_info"])
                diagnostics_rows.append(diag)

        protocol_summary = {
            "group_source": group_source,
            "n_unique_groups": int(groups.nunique()),
            "methods": {},
        }
        for name in method_names:
            method_summary = summarize(protocol_results[name])
            split_summary = summarize_split_info(protocol_results[name])
            protocol_summary["methods"][name] = method_summary
            protocol_summary["methods"][name]["split_info_summary"] = split_summary

            main_row = {
                "protocol": protocol,
                "group_source": group_source,
                "method": name,
            }
            for metric in ["roc_auc", "pr_auc", "f1", "balanced_acc"]:
                if metric in method_summary:
                    main_row[f"{metric}_mean"] = method_summary[metric]["mean"]
                    main_row[f"{metric}_std"] = method_summary[metric]["std"]
                else:
                    main_row[f"{metric}_mean"] = np.nan
                    main_row[f"{metric}_std"] = np.nan
            main_rows.append(main_row)

            if "brier" in method_summary or "ece" in method_summary:
                calibration_rows.append({
                    "protocol": protocol,
                    "group_source": group_source,
                    "method": name,
                    "brier_mean": method_summary.get("brier", {}).get("mean", np.nan),
                    "brier_std": method_summary.get("brier", {}).get("std", np.nan),
                    "ece_mean": method_summary.get("ece", {}).get("mean", np.nan),
                    "ece_std": method_summary.get("ece", {}).get("std", np.nan),
                })

        summary["protocol_results"][protocol] = protocol_summary

    # Save outputs
    with open(os.path.join(args.outdir, "section2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(main_rows).to_csv(os.path.join(args.outdir, "section2_main_table.csv"), index=False)
    pd.DataFrame(per_run_rows).to_csv(os.path.join(args.outdir, "section2_per_run_metrics.csv"), index=False)
    pd.DataFrame(diagnostics_rows).to_csv(os.path.join(args.outdir, "section2_split_diagnostics.csv"), index=False)
    pd.DataFrame(calibration_rows).to_csv(os.path.join(args.outdir, "section2_calibration.csv"), index=False)

    # Supplementary long table
    supp_rows = []
    for protocol, p_summary in summary["protocol_results"].items():
        group_source = p_summary["group_source"]
        for method, method_summary in p_summary["methods"].items():
            for metric, metric_val in method_summary.items():
                if metric == "split_info_summary":
                    continue
                supp_rows.append({
                    "protocol": protocol,
                    "group_source": group_source,
                    "method": method,
                    "metric": metric,
                    "mean": metric_val["mean"],
                    "std": metric_val["std"],
                })
    pd.DataFrame(supp_rows).to_csv(os.path.join(args.outdir, "section2_supplementary_table.csv"), index=False)

    with open(os.path.join(args.outdir, "section2_feature_sets.json"), "w", encoding="utf-8") as f:
        json.dump(feature_sets, f, ensure_ascii=False, indent=2)

    print("===== Section 2 robustness experiment completed =====")
    print(json.dumps(summary["metadata"], ensure_ascii=False, indent=2))
    print(f"Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()
