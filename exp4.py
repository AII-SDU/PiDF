#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4 script for:
Label-Efficient Discovery with Active Learning

This script is built on top of the user's current benchmark framework and the
Section-1/Section-2/Section-3 logic.

It provides two experiment families:

1) Strategy comparison on the PiDF representation
   - random
   - uncertainty
   - potential
   - diversity
   - hybrid strategies with user-specified weights

2) Representation comparison under a selected acquisition strategy
   - Conventional base + ML
   - Weighted TF + ML
   - PiDF + ML

Main design choices:
- deduplication is supported and recommended
- an outer train-pool / test split is used for evaluation
- active learning is performed only on the outer train pool
- the outer test set is fixed for each seed
- an inner train/val split is built from the current labeled set at each round
  for model fitting and threshold tuning
- metrics are tracked as learning curves versus labeling budget
- cumulative near-stable discoveries are tracked in the queried set and in the
  full labeled set
- AULC summaries are computed for key metrics
- main-paper outputs focus on the trade-off between predictive performance and
  near-stable candidate enrichment
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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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


def choose_group_column(df: pd.DataFrame) -> str:
    for col in ["host_id", "chemsys_query", "parent_material_id"]:
        if col in df.columns:
            return col
    raise ValueError("group split requires one of: host_id / chemsys_query / parent_material_id")


def build_label_column(df: pd.DataFrame, label_mode: str, ehull_threshold: float) -> pd.DataFrame:
    df = df.copy()

    if label_mode == "label":
        if "label" not in df.columns:
            raise ValueError("CSV is missing column: label")
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        return df

    if label_mode == "is_stable":
        if "is_stable" not in df.columns:
            raise ValueError("CSV is missing column: is_stable")
        df = df[df["is_stable"].notna()].copy()
        df["label"] = df["is_stable"].astype(int)
        return df

    if label_mode == "ehull":
        if "energy_above_hull" not in df.columns:
            raise ValueError("CSV is missing column: energy_above_hull")
        df = df[df["energy_above_hull"].notna()].copy()
        df["label"] = (df["energy_above_hull"].astype(float) <= ehull_threshold).astype(int)
        return df

    raise ValueError("label_mode must be one of: label / is_stable / ehull")


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
# Feature engineering
# =========================
def enrich_composition_columns(df: pd.DataFrame) -> pd.DataFrame:
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

    Floating-point amounts such as 0.3 and 0.30000000000000004 are rounded
    before serialization so that equivalent normalized compositions are
    deduplicated consistently.
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
        raise ValueError(f"dedup_key={dedup_key} is not present in the dataframe")

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


def make_outer_split(
    df: pd.DataFrame,
    split_mode: str,
    seed: int,
    test_size: float = 0.2,
    max_group_trials: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(df))
    y = df["label"].values

    if split_mode == "random":
        train_pool_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
        return train_pool_idx, test_idx

    if split_mode == "group":
        group_col = choose_group_column(df)
        groups = df[group_col].values
        for trial in range(max_group_trials):
            rseed = seed + 1000 * trial
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rseed)
            train_pool_idx, test_idx = next(splitter.split(indices, y, groups=groups))
            if _all_subsets_have_both_classes(y, [train_pool_idx, test_idx]):
                return train_pool_idx, test_idx
        raise RuntimeError(
            f"group split failed to produce pool/test containing both classes after {max_group_trials} trials"
        )

    raise ValueError("split_mode must be one of: random / group")


def make_inner_train_val_split(
    y: np.ndarray,
    seed: int,
    val_frac: float = 0.2,
    min_val: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    n = len(indices)
    if n < 6:
        # fallback: use all data for both train and val-like selection; caller should handle cautiously
        return indices, indices

    test_size = max(min_val / n, val_frac)
    test_size = min(max(test_size, 0.1), 0.4)

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
        if _all_subsets_have_both_classes(y, [train_idx, val_idx]):
            return train_idx, val_idx
    except Exception:
        pass

    for trial in range(50):
        rseed = seed + 1000 * trial
        perm = np.random.RandomState(rseed).permutation(indices)
        n_val = int(round(test_size * n))
        n_val = max(min_val, n_val)
        n_val = min(n - 2, n_val)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        if len(train_idx) == 0:
            continue
        if _all_subsets_have_both_classes(y, [train_idx, val_idx]):
            return train_idx, val_idx

    return indices, indices


def choose_initial_labeled(
    y_pool: np.ndarray,
    initial_ratio: float,
    initial_size: Optional[int],
    seed: int,
    min_per_class: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a small stratified initial labeled set for oracle-labeling simulation.
    """
    n_pool = len(y_pool)
    if initial_size is None:
        n_init = max(2 * min_per_class, int(round(n_pool * initial_ratio)))
    else:
        n_init = int(initial_size)
    n_init = min(max(n_init, 2 * min_per_class), n_pool - 1)

    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_pool == 1)[0]
    neg_idx = np.where(y_pool == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Pool must contain both classes for active learning.")

    n_pos = int(round(n_init * len(pos_idx) / n_pool))
    n_pos = min(max(min_per_class, n_pos), len(pos_idx))
    n_neg = min(max(min_per_class, n_init - n_pos), len(neg_idx))

    # Fill any remaining slots without replacement while preserving at least one
    # example from each class.
    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    chosen = list(chosen_pos.tolist()) + list(chosen_neg.tolist())

    if len(chosen) < n_init:
        remaining = np.array([i for i in range(n_pool) if i not in set(chosen)], dtype=int)
        n_extra = min(n_init - len(chosen), len(remaining))
        if n_extra > 0:
            chosen.extend(rng.choice(remaining, size=n_extra, replace=False).tolist())

    labeled_rel = np.sort(np.array(chosen, dtype=int))
    unlabeled_rel = np.array([i for i in range(n_pool) if i not in set(labeled_rel.tolist())], dtype=int)
    return labeled_rel, unlabeled_rel


# =========================
# Models and metrics
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
    model_type: str = "logreg"  # logreg / mlp / rf
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
    X_target: np.ndarray,
    cfg: TrainConfig,
):
    if cfg.model_type == "mlp":
        model = train_mlp(X_train, y_train, X_val, y_val, cfg)
        val_probs = predict_prob_mlp(model, X_val, cfg.device)
        target_probs = predict_prob_mlp(model, X_target, cfg.device)
        return model, val_probs, target_probs

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
        target_probs = model.predict_proba(X_target)[:, 1]
        return model, val_probs, target_probs

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
        target_probs = model.predict_proba(X_target)[:, 1]
        return model, val_probs, target_probs

    raise ValueError("model_type must be one of: mlp / logreg / rf")


def predict_with_model(model, X: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    if cfg.model_type == "mlp":
        return predict_prob_mlp(model, X, cfg.device)
    return model.predict_proba(X)[:, 1]


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Binary expected calibration error.

    For each probability bin, compare the empirical positive rate with the mean
    predicted probability. This is better aligned with probability calibration
    than comparing hard-threshold accuracy with confidence.
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
    probs: np.ndarray,
    threshold: float,
    include_calibration: bool = True,
) -> Dict[str, float]:
    pred = (probs >= threshold).astype(int)
    out: Dict[str, float] = {}
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
    if include_calibration:
        try:
            out["brier"] = float(brier_score_loss(y_true, probs))
        except ValueError:
            out["brier"] = float("nan")
        out["ece"] = compute_ece(y_true, probs)
    return out


# =========================
# Feature sets
# =========================
def _ordered_unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def get_conventional_base_spec(df: pd.DataFrame) -> Dict[str, List[str]]:
    frac_num = sorted([c for c in df.columns if c.startswith("frac_")])
    conventional_cat = [c for c in ["mix_site", "A_host", "A_sub", "B_host", "B_sub"] if c in df.columns]
    identity_num = [c for c in ["mix_ratio", "n_A_species", "n_B_species"] if c in df.columns]
    raw_num = [c for c in ["r_A_eff", "r_B_eff", "q_A_eff", "q_B_eff"] if c in df.columns]
    conventional_num = _ordered_unique(identity_num + frac_num + raw_num)
    return {
        "categorical": conventional_cat,
        "numeric": conventional_num,
    }


def get_representation_specs(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    base = get_conventional_base_spec(df)
    base_cat = list(base["categorical"])
    base_num = list(base["numeric"])
    specs = {
        "Conventional Base + ML": {
            "categorical": base_cat,
            "numeric": base_num,
        },
        "Weighted TF + ML": {
            "categorical": base_cat,
            "numeric": _ordered_unique(base_num + [c for c in ["wtf"] if c in df.columns]),
        },
        "PiDF + ML": {
            "categorical": base_cat,
            "numeric": _ordered_unique(base_num + [c for c in ["wtf", "sigma_A", "sigma_B", "delta_q"] if c in df.columns]),
        },
    }
    return specs


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
# Fitting bundle on current labeled set
# =========================
def fit_model_bundle(
    labeled_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spec: Dict[str, List[str]],
    cfg: TrainConfig,
    inner_val_frac: float,
    seed: int,
) -> Dict[str, Any]:
    cat_cols = list(spec["categorical"])
    num_cols = list(spec["numeric"])
    used_cols = cat_cols + num_cols

    y_labeled = labeled_df["label"].values.astype(int)
    train_rel, val_rel = make_inner_train_val_split(y_labeled, seed=seed, val_frac=inner_val_frac)

    train_df = labeled_df.iloc[train_rel].reset_index(drop=True)
    val_df = labeled_df.iloc[val_rel].reset_index(drop=True)

    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train = preprocessor.fit_transform(train_df[used_cols])
    X_val = preprocessor.transform(val_df[used_cols])
    X_test = preprocessor.transform(test_df[used_cols])

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()

    y_train = train_df["label"].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.float32)
    y_test = test_df["label"].values.astype(int)

    model, val_probs, test_probs = fit_predict_model(X_train, y_train, X_val, y_val, X_test, cfg)
    thr, best_val_score = tune_probability_threshold(y_val.astype(int), val_probs, cfg.threshold_objective)
    test_metrics = compute_metrics(y_true=y_test, probs=test_probs, threshold=thr)
    test_metrics["val_threshold_best"] = float(thr)
    test_metrics[f"val_{cfg.threshold_objective}_best"] = float(best_val_score)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "threshold": float(thr),
        "test_metrics": test_metrics,
        "test_probs": test_probs,
    }


# =========================
# Acquisition utilities
# =========================
def normalize_01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x = np.asarray(x, dtype=float)
    xmin = np.min(x)
    xmax = np.max(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def min_distances(X_u: np.ndarray, X_l: np.ndarray) -> np.ndarray:
    if len(X_u) == 0:
        return np.array([], dtype=float)
    if len(X_l) == 0:
        return np.ones(len(X_u), dtype=float)
    # small dataset; dense pairwise distance is acceptable
    diff = X_u[:, None, :] - X_l[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.sqrt(np.min(dist2, axis=1))


PIDF_DIVERSITY_COLS = ["wtf", "sigma_A", "sigma_B", "delta_q"]


def build_pidf_diversity_matrices(
    pool_df: pd.DataFrame,
    labeled_rel: np.ndarray,
    unlabeled_rel: np.ndarray,
    cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build standardized PiDF descriptor matrices for diversity acquisition.

    The scaler is fitted on the current train-pool candidates only; no label
    information is used. This makes the diversity term consistent with the
    method definition D_t(c) based on standardized PiDF descriptor vectors.
    """
    if cols is None:
        cols = PIDF_DIVERSITY_COLS

    missing = [c for c in cols if c not in pool_df.columns]
    if missing:
        raise ValueError(f"Missing PiDF diversity columns: {missing}")

    X_all = pool_df[cols].astype(float).values
    scaler = StandardScaler()
    X_all_std = scaler.fit_transform(X_all)

    return X_all_std[unlabeled_rel], X_all_std[labeled_rel]


def parse_strategy_weights(strategy_name: str) -> Optional[Tuple[float, float, float]]:
    if not strategy_name.startswith("hybrid_"):
        return None
    parts = strategy_name.split("_")
    if len(parts) != 4:
        raise ValueError(f"Malformed hybrid strategy name: {strategy_name}")
    return float(parts[1]), float(parts[2]), float(parts[3])


def build_strategy_list() -> List[str]:
    return [
        "random",
        "uncertainty",
        "potential",
        "diversity",
        "hybrid_0.15_0.55_0.30",
        "hybrid_0.20_0.50_0.30",
        "hybrid_0.25_0.45_0.30",
    ]


def select_query_batch(
    strategy: str,
    probs_u: np.ndarray,
    X_u: np.ndarray,
    X_l: np.ndarray,
    query_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    n = len(probs_u)
    if n == 0:
        return np.array([], dtype=int)
    q = min(query_size, n)

    if strategy == "random":
        return rng.choice(np.arange(n), size=q, replace=False)

    uncertainty = 1.0 - 2.0 * np.abs(probs_u - 0.5)
    potential = probs_u.copy()
    diversity = min_distances(X_u, X_l)

    if strategy == "uncertainty":
        score = uncertainty
    elif strategy == "potential":
        score = potential
    elif strategy == "diversity":
        score = diversity
    else:
        weights = parse_strategy_weights(strategy)
        if weights is None:
            raise ValueError(f"Unknown strategy: {strategy}")
        wu, wp, wd = weights
        score = wu * normalize_01(uncertainty) + wp * normalize_01(potential) + wd * normalize_01(diversity)

    # stable sort with small random jitter for deterministic tie-breaking under rng
    jitter = rng.uniform(low=0.0, high=1e-9, size=n)
    order = np.argsort(-(score + jitter), kind="mergesort")
    return order[:q]


# =========================
# Active-learning core
# =========================
def run_active_learning_single(
    df: pd.DataFrame,
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    initial_rel_idx: np.ndarray,
    strategy: str,
    spec: Dict[str, List[str]],
    cfg: TrainConfig,
    seed: int,
    query_size: int,
    n_rounds: int,
    inner_val_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed + 12345)

    pool_df = df.iloc[train_pool_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    labeled_rel = np.array(sorted(initial_rel_idx.tolist()), dtype=int)
    unlabeled_rel = np.array([i for i in range(len(pool_df)) if i not in set(labeled_rel.tolist())], dtype=int)

    curve_records: List[Dict[str, Any]] = []
    query_records: List[Dict[str, Any]] = []
    cumulative_new_stable = 0

    for round_id in range(n_rounds + 1):
        labeled_df = pool_df.iloc[labeled_rel].reset_index(drop=True)
        bundle = fit_model_bundle(
            labeled_df=labeled_df,
            test_df=test_df,
            spec=spec,
            cfg=cfg,
            inner_val_frac=inner_val_frac,
            seed=seed + 100 * round_id,
        )

        test_metrics = dict(bundle["test_metrics"])
        budget = int(len(labeled_rel))
        total_stable_in_labeled = int(pool_df.iloc[labeled_rel]["label"].sum())
        total_queried = int(max(0, budget - len(initial_rel_idx)))

        curve_rec = {
            "seed": int(seed),
            "round": int(round_id),
            "budget": budget,
            "strategy": strategy,
            "stable_discoveries_total": total_stable_in_labeled,
            "stable_discoveries_new": int(cumulative_new_stable),
            "queried_total": total_queried,
        }
        curve_rec.update(test_metrics)
        curve_records.append(curve_rec)

        if round_id == n_rounds or len(unlabeled_rel) == 0:
            break

        preprocessor = bundle["preprocessor"]
        model = bundle["model"]
        used_cols = list(spec["categorical"]) + list(spec["numeric"])

        # Model-space features are used for probability prediction.
        X_l_model = preprocessor.transform(pool_df.iloc[labeled_rel][used_cols])
        X_u_model = preprocessor.transform(pool_df.iloc[unlabeled_rel][used_cols])
        if hasattr(X_l_model, "toarray"):
            X_l_model = X_l_model.toarray()
            X_u_model = X_u_model.toarray()

        probs_u = predict_with_model(model, X_u_model, cfg)

        # Diversity is computed in standardized PiDF descriptor space, not in
        # the full one-hot/composition feature space.
        X_u_div, X_l_div = build_pidf_diversity_matrices(
            pool_df=pool_df,
            labeled_rel=labeled_rel,
            unlabeled_rel=unlabeled_rel,
        )

        chosen_rel_in_u = select_query_batch(
            strategy=strategy,
            probs_u=probs_u,
            X_u=X_u_div,
            X_l=X_l_div,
            query_size=query_size,
            rng=rng,
        )
        chosen_pool_rel = unlabeled_rel[chosen_rel_in_u]
        chosen_labels = pool_df.iloc[chosen_pool_rel]["label"].values.astype(int)
        cumulative_new_stable += int(chosen_labels.sum())

        for pos, (pool_rel, lab, prob) in enumerate(zip(chosen_pool_rel.tolist(), chosen_labels.tolist(), probs_u[chosen_rel_in_u].tolist())):
            query_records.append({
                "seed": int(seed),
                "round": int(round_id + 1),
                "strategy": strategy,
                "acq_rank": int(pos + 1),
                "pool_rel_idx": int(pool_rel),
                "formula_pretty": pool_df.iloc[pool_rel].get("formula_pretty", None),
                "label": int(lab),
                "pred_prob_before_query": float(prob),
            })

        labeled_rel = np.array(sorted(np.concatenate([labeled_rel, chosen_pool_rel])), dtype=int)
        keep_mask = np.ones(len(unlabeled_rel), dtype=bool)
        keep_mask[chosen_rel_in_u] = False
        unlabeled_rel = unlabeled_rel[keep_mask]

    return pd.DataFrame(curve_records), pd.DataFrame(query_records)


# =========================
# Aggregation / AULC / plots
# =========================
def aggregate_curves(curves_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    numeric_cols = [
        "budget", "roc_auc", "pr_auc", "f1", "balanced_acc", "precision", "recall",
        "brier", "ece", "stable_discoveries_total", "stable_discoveries_new", "queried_total",
    ]
    rows = []
    for keys, sub in curves_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {k: v for k, v in zip(group_cols, keys)}
        base["n_runs"] = int(sub["seed"].nunique())
        for col in numeric_cols:
            if col not in sub.columns:
                continue
            vals = sub[col].astype(float).values
            base[f"{col}_mean"] = float(np.nanmean(vals))
            base[f"{col}_std"] = float(np.nanstd(vals))
        rows.append(base)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in group_cols + ["budget"] if c in out.columns]
    if len(sort_cols) > 0:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def compute_aulc_per_seed(curves_df: pd.DataFrame, entity_col: str, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for (entity, seed), sub in curves_df.groupby([entity_col, "seed"]):
        sub = sub.sort_values("budget")
        x = sub["budget"].values.astype(float)
        rec = {entity_col: entity, "seed": int(seed)}
        for m in metrics:
            y = sub[m].values.astype(float)
            if len(x) < 2:
                area = float(y[0]) if len(y) == 1 else float("nan")
            else:
                area = float(np.trapz(y, x) / (x[-1] - x[0]))
            rec[f"aulc_{m}"] = area
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_aulc(aulc_df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    rows = []
    metric_cols = [c for c in aulc_df.columns if c.startswith("aulc_")]
    for entity, sub in aulc_df.groupby(entity_col):
        rec = {entity_col: entity, "n_runs": int(sub["seed"].nunique())}
        for c in metric_cols:
            vals = sub[c].astype(float).values
            rec[f"{c}_mean"] = float(np.nanmean(vals))
            rec[f"{c}_std"] = float(np.nanstd(vals))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(entity_col).reset_index(drop=True)


def get_main_strategy_order(selected_strategy: str) -> List[str]:
    """
    Strategies to be shown in the main paper.

    Additional hybrid weight variants are kept in the raw output files but are
    excluded from the main table/figure to avoid clutter.
    """
    return ["random", "uncertainty", "potential", "diversity", selected_strategy]


def strategy_display_name(strategy: str, selected_strategy: str) -> str:
    if strategy == selected_strategy and strategy.startswith("hybrid_"):
        return "Hybrid"
    mapping = {
        "random": "Random",
        "uncertainty": "Uncertainty",
        "potential": "Potential",
        "diversity": "Diversity",
    }
    return mapping.get(strategy, strategy)


def format_mean_std(mean: float, std: float, ndigits: int = 3) -> str:
    if pd.isna(mean):
        return "---"
    return f"{mean:.{ndigits}f}±{std:.{ndigits}f}"


def build_main_strategy_tradeoff_table(
    strategy_final: pd.DataFrame,
    strategy_aulc: pd.DataFrame,
    selected_strategy: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the compact main-table output for Section 4.

    The table focuses on the trade-off between predictive performance and
    near-stable candidate enrichment:
      - final PR-AUC, balanced accuracy, and F1;
      - final cumulative near-stable discoveries;
      - AULC PR-AUC;
      - AULC stable discoveries.
    """
    main_order = get_main_strategy_order(selected_strategy)
    rows_raw = []
    rows_fmt = []

    final_idx = {str(r["strategy"]): r for _, r in strategy_final.iterrows()}
    aulc_idx = {str(r["strategy"]): r for _, r in strategy_aulc.iterrows()}

    for strategy in main_order:
        if strategy not in final_idx or strategy not in aulc_idx:
            continue

        f = final_idx[strategy]
        a = aulc_idx[strategy]
        name = strategy_display_name(strategy, selected_strategy)

        raw = {
            "strategy": strategy,
            "display_name": name,
            "final_pr_auc_mean": float(f["pr_auc_mean"]),
            "final_pr_auc_std": float(f["pr_auc_std"]),
            "final_balanced_acc_mean": float(f["balanced_acc_mean"]),
            "final_balanced_acc_std": float(f["balanced_acc_std"]),
            "final_f1_mean": float(f["f1_mean"]),
            "final_f1_std": float(f["f1_std"]),
            "final_stable_discoveries_mean": float(f["stable_discoveries_total_mean"]),
            "final_stable_discoveries_std": float(f["stable_discoveries_total_std"]),
            "aulc_pr_auc_mean": float(a["aulc_pr_auc_mean"]),
            "aulc_pr_auc_std": float(a["aulc_pr_auc_std"]),
            "aulc_stable_discoveries_mean": float(a["aulc_stable_discoveries_total_mean"]),
            "aulc_stable_discoveries_std": float(a["aulc_stable_discoveries_total_std"]),
        }
        rows_raw.append(raw)

        rows_fmt.append({
            "Strategy": name,
            "Final PR-AUC": format_mean_std(raw["final_pr_auc_mean"], raw["final_pr_auc_std"]),
            "Final Bal. Acc.": format_mean_std(raw["final_balanced_acc_mean"], raw["final_balanced_acc_std"]),
            "Final F1": format_mean_std(raw["final_f1_mean"], raw["final_f1_std"]),
            "Stable Disc.": format_mean_std(raw["final_stable_discoveries_mean"], raw["final_stable_discoveries_std"], ndigits=1),
            "AULC PR-AUC": format_mean_std(raw["aulc_pr_auc_mean"], raw["aulc_pr_auc_std"]),
            "AULC Stable Disc.": format_mean_std(raw["aulc_stable_discoveries_mean"], raw["aulc_stable_discoveries_std"], ndigits=1),
        })

    return pd.DataFrame(rows_raw), pd.DataFrame(rows_fmt)


def filter_main_strategy_curve(
    strategy_curve_mean: pd.DataFrame,
    selected_strategy: str,
) -> pd.DataFrame:
    main_order = get_main_strategy_order(selected_strategy)
    out = strategy_curve_mean[strategy_curve_mean["strategy"].isin(main_order)].copy()
    out["display_name"] = out["strategy"].apply(lambda s: strategy_display_name(str(s), selected_strategy))
    out["strategy_order"] = out["strategy"].apply(lambda s: main_order.index(str(s)) if str(s) in main_order else 999)
    out = out.sort_values(["strategy_order", "budget"]).reset_index(drop=True)
    return out


def plot_main_stable_discoveries(
    strategy_curve_mean: pd.DataFrame,
    selected_strategy: str,
    outdir: str,
) -> None:
    """
    Main Section-4 figure: cumulative near-stable discoveries vs labeling budget.
    Only the four single-objective strategies and the fixed hybrid strategy are
    shown.
    """
    plot_df = filter_main_strategy_curve(strategy_curve_mean, selected_strategy)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, sub in plot_df.groupby("display_name", sort=False):
        sub = sub.sort_values("budget")
        x = sub["budget"].values.astype(float)
        y = sub["stable_discoveries_total_mean"].values.astype(float)
        yerr = sub["stable_discoveries_total_std"].values.astype(float)
        ax.plot(x, y, marker="o", linewidth=1.6, markersize=4.5, label=str(name))
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15)

    ax.set_xlabel("Labeling budget")
    ax.set_ylabel("Cumulative near-stable discoveries")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    png_path = os.path.join(outdir, "fig4_stable_discoveries_main.png")
    pdf_path = os.path.join(outdir, "fig4_stable_discoveries_main.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def choose_best_hybrid(final_df: pd.DataFrame) -> str:
    hybrids = final_df[final_df["strategy"].str.startswith("hybrid_")].copy()
    if len(hybrids) == 0:
        raise ValueError("No hybrid strategies available to choose from.")
    hybrids = hybrids.sort_values(
        by=["pr_auc_mean", "stable_discoveries_total_mean", "roc_auc_mean"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    return str(hybrids.iloc[0]["strategy"])


def plot_learning_curves(
    agg_df: pd.DataFrame,
    entity_col: str,
    metric: str,
    title: str,
    save_path: str,
) -> None:
    """
    Supplementary learning-curve plot. The main-paper Section-4 figure is
    generated by plot_main_stable_discoveries().
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"
    for entity, sub in agg_df.groupby(entity_col):
        sub = sub.sort_values("budget")
        x = sub["budget"].values.astype(float)
        y = sub[metric_mean].values.astype(float)
        yerr = sub[metric_std].values.astype(float)
        ax.plot(x, y, marker="o", label=str(entity))
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15)
    ax.set_xlabel("Labeling budget")
    ax.set_ylabel(metric.replace("_", " ").title())
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path.lower().endswith(".png"):
        fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./mp_data/exp42_perovskite_like_subset.csv")
    parser.add_argument("--outdir", type=str, default="runs/section4_active_learning")
    parser.add_argument("--split", type=str, default="random", choices=["random", "group"])
    parser.add_argument("--dedup-key", type=str, default="auto", choices=["none", "auto", "normalized_composition_canonical", "formula_pretty"])
    parser.add_argument("--label-mode", type=str, default="ehull", choices=["label", "is_stable", "ehull"])
    parser.add_argument("--ehull-threshold", type=float, default=0.05)
    parser.add_argument("--model-type", type=str, default="logreg", choices=["logreg", "mlp", "rf"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--initial-ratio", type=float, default=0.1)
    parser.add_argument("--initial-size", type=int, default=20)
    parser.add_argument("--query-size", type=int, default=10)
    parser.add_argument("--n-rounds", type=int, default=8)
    parser.add_argument("--inner-val-frac", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--selected-strategy",
        type=str,
        default="hybrid_0.25_0.45_0.30",
        help="A fixed strategy for representation comparison. Use 'auto' only for exploratory analysis.",
    )
    parser.add_argument("--threshold-objective", type=str, default="balanced_acc", choices=["balanced_acc", "f1"])
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--save-query-history", action="store_true")
    parser.add_argument("--save-clean-dataset", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    set_seed(0)

    # -------------------------
    # Load and prepare dataset
    # -------------------------
    raw_df = pd.read_csv(args.csv)
    duplicate_stats_before = {
        "n_duplicate_normalized_composition": int(raw_df["normalized_composition"].duplicated().sum()) if "normalized_composition" in raw_df.columns else 0,
        "n_duplicate_formula_pretty": int(raw_df["formula_pretty"].duplicated().sum()) if "formula_pretty" in raw_df.columns else 0,
    }

    df = build_label_column(raw_df, args.label_mode, args.ehull_threshold)
    df = enrich_composition_columns(df)
    df = enrich_fraction_features(df)
    df = canonicalize_normalized_composition(df)

    before_dedup = len(df)
    df = deduplicate_dataset(df, args.dedup_key)
    after_dedup = len(df)

    # basic checks
    required_descriptor_cols = ["wtf", "sigma_A", "sigma_B", "delta_q"]
    missing = [c for c in required_descriptor_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required descriptor columns: {missing}")

    before_descriptor_filter = len(df)
    df = df[df[required_descriptor_cols].notna().all(axis=1)].copy().reset_index(drop=True)
    after_descriptor_filter = len(df)

    if len(df) == 0:
        raise ValueError("No samples remain after preprocessing.")
    if df["label"].nunique() < 2:
        raise ValueError("The dataset must contain both classes.")

    if args.save_clean_dataset:
        df.to_csv(os.path.join(args.outdir, "clean_dataset.csv"), index=False)

    specs = get_representation_specs(df)
    strategy_list = build_strategy_list()

    cfg = TrainConfig(
        model_type=args.model_type,
        device=args.device,
        threshold_objective=args.threshold_objective,
        logreg_c=args.logreg_c,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
    )

    # -------------------------
    # Strategy comparison on PiDF
    # -------------------------
    pidf_spec = specs["PiDF + ML"]
    strategy_curve_frames = []
    strategy_query_frames = []

    split_records = []

    for seed in args.seeds:
        set_seed(seed)
        train_pool_idx, test_idx = make_outer_split(df, split_mode=args.split, seed=seed, test_size=args.test_size)

        y_pool = df.iloc[train_pool_idx]["label"].values.astype(int)
        initial_rel, _ = choose_initial_labeled(
            y_pool=y_pool,
            initial_ratio=args.initial_ratio,
            initial_size=args.initial_size,
            seed=seed,
            min_per_class=2,
        )

        split_records.append({
            "seed": int(seed),
            "n_train_pool": int(len(train_pool_idx)),
            "n_test": int(len(test_idx)),
            "initial_labeled": int(len(initial_rel)),
            "final_budget": int(len(initial_rel) + args.n_rounds * args.query_size),
            "n_test_stable": int(df.iloc[test_idx]["label"].sum()),
            "n_test_unstable": int(len(test_idx) - df.iloc[test_idx]["label"].sum()),
        })

        for strategy in strategy_list:
            curve_df, query_df = run_active_learning_single(
                df=df,
                train_pool_idx=train_pool_idx,
                test_idx=test_idx,
                initial_rel_idx=initial_rel,
                strategy=strategy,
                spec=pidf_spec,
                cfg=cfg,
                seed=seed,
                query_size=args.query_size,
                n_rounds=args.n_rounds,
                inner_val_frac=args.inner_val_frac,
            )
            strategy_curve_frames.append(curve_df)
            if args.save_query_history:
                strategy_query_frames.append(query_df)

    strategy_curves = pd.concat(strategy_curve_frames, ignore_index=True)
    strategy_curve_mean = aggregate_curves(strategy_curves, group_cols=["strategy", "budget"])
    strategy_final = strategy_curve_mean.groupby("strategy", as_index=False).tail(1).reset_index(drop=True)
    strategy_aulc_seed = compute_aulc_per_seed(
        strategy_curves,
        entity_col="strategy",
        metrics=["roc_auc", "pr_auc", "balanced_acc", "stable_discoveries_total", "stable_discoveries_new"],
    )
    strategy_aulc = summarize_aulc(strategy_aulc_seed, entity_col="strategy")

    if args.selected_strategy == "auto":
        selected_strategy = choose_best_hybrid(strategy_final)
    else:
        selected_strategy = args.selected_strategy

    # Compact main-paper outputs for Section 4:
    #   Fig. 4: cumulative near-stable discoveries;
    #   Table IV: final-budget and AULC trade-off comparison.
    strategy_tradeoff_raw, strategy_tradeoff_formatted = build_main_strategy_tradeoff_table(
        strategy_final=strategy_final,
        strategy_aulc=strategy_aulc,
        selected_strategy=selected_strategy,
    )
    main_strategy_curve = filter_main_strategy_curve(strategy_curve_mean, selected_strategy)

    # -------------------------
    # Representation comparison under selected strategy
    # -------------------------
    rep_curve_frames = []
    rep_query_frames = []

    for seed in args.seeds:
        set_seed(seed)
        train_pool_idx, test_idx = make_outer_split(df, split_mode=args.split, seed=seed, test_size=args.test_size)
        y_pool = df.iloc[train_pool_idx]["label"].values.astype(int)
        initial_rel, _ = choose_initial_labeled(
            y_pool=y_pool,
            initial_ratio=args.initial_ratio,
            initial_size=args.initial_size,
            seed=seed,
            min_per_class=2,
        )

        for rep_name, spec in specs.items():
            curve_df, query_df = run_active_learning_single(
                df=df,
                train_pool_idx=train_pool_idx,
                test_idx=test_idx,
                initial_rel_idx=initial_rel,
                strategy=selected_strategy,
                spec=spec,
                cfg=cfg,
                seed=seed,
                query_size=args.query_size,
                n_rounds=args.n_rounds,
                inner_val_frac=args.inner_val_frac,
            )
            curve_df["representation"] = rep_name
            rep_curve_frames.append(curve_df)
            if args.save_query_history:
                query_df["representation"] = rep_name
                rep_query_frames.append(query_df)

    rep_curves = pd.concat(rep_curve_frames, ignore_index=True)
    rep_curve_mean = aggregate_curves(rep_curves, group_cols=["representation", "budget"])
    rep_final = rep_curve_mean.groupby("representation", as_index=False).tail(1).reset_index(drop=True)
    rep_aulc_seed = compute_aulc_per_seed(
        rep_curves,
        entity_col="representation",
        metrics=["roc_auc", "pr_auc", "balanced_acc", "stable_discoveries_total", "stable_discoveries_new"],
    )
    rep_aulc = summarize_aulc(rep_aulc_seed, entity_col="representation")

    # -------------------------
    # Save outputs
    # -------------------------
    strategy_curves.to_csv(os.path.join(args.outdir, "strategy_curves_per_seed.csv"), index=False)
    strategy_curve_mean.to_csv(os.path.join(args.outdir, "strategy_curve_mean.csv"), index=False)
    strategy_final.to_csv(os.path.join(args.outdir, "strategy_final_budget.csv"), index=False)
    strategy_aulc_seed.to_csv(os.path.join(args.outdir, "strategy_aulc_per_seed.csv"), index=False)
    strategy_aulc.to_csv(os.path.join(args.outdir, "strategy_aulc_summary.csv"), index=False)

    # Main-paper Section-4 outputs.
    main_strategy_curve.to_csv(os.path.join(args.outdir, "main_strategy_curve.csv"), index=False)
    strategy_tradeoff_raw.to_csv(os.path.join(args.outdir, "table4_active_learning_tradeoff_raw.csv"), index=False)
    strategy_tradeoff_formatted.to_csv(os.path.join(args.outdir, "table4_active_learning_tradeoff_formatted.csv"), index=False)

    rep_curves.to_csv(os.path.join(args.outdir, "representation_curves_per_seed.csv"), index=False)
    rep_curve_mean.to_csv(os.path.join(args.outdir, "representation_curve_mean.csv"), index=False)
    rep_final.to_csv(os.path.join(args.outdir, "representation_final_budget.csv"), index=False)
    rep_aulc_seed.to_csv(os.path.join(args.outdir, "representation_aulc_per_seed.csv"), index=False)
    rep_aulc.to_csv(os.path.join(args.outdir, "representation_aulc_summary.csv"), index=False)

    pd.DataFrame(split_records).to_csv(os.path.join(args.outdir, "split_budget_diagnostics.csv"), index=False)

    if args.save_query_history and len(strategy_query_frames) > 0:
        pd.concat(strategy_query_frames, ignore_index=True).to_csv(
            os.path.join(args.outdir, "strategy_query_history.csv"), index=False
        )
    if args.save_query_history and len(rep_query_frames) > 0:
        pd.concat(rep_query_frames, ignore_index=True).to_csv(
            os.path.join(args.outdir, "representation_query_history.csv"), index=False
        )

    # Main-paper figure: cumulative near-stable discoveries.
    plot_main_stable_discoveries(
        strategy_curve_mean=strategy_curve_mean,
        selected_strategy=selected_strategy,
        outdir=args.outdir,
    )

    # Supplementary diagnostic plots. These are useful for checking the
    # predictive-learning trade-off, but they are not the recommended main
    # figure for Section 4.
    plot_learning_curves(
        strategy_curve_mean, entity_col="strategy", metric="pr_auc",
        title="Supplementary: strategy comparison on PiDF (PR-AUC)",
        save_path=os.path.join(args.outdir, "supp_strategy_pr_auc_curve_all.png"),
    )
    plot_learning_curves(
        strategy_curve_mean, entity_col="strategy", metric="balanced_acc",
        title="Supplementary: strategy comparison on PiDF (balanced accuracy)",
        save_path=os.path.join(args.outdir, "supp_strategy_balanced_acc_curve_all.png"),
    )
    plot_learning_curves(
        rep_curve_mean, entity_col="representation", metric="pr_auc",
        title=f"Supplementary: representation comparison under {selected_strategy} (PR-AUC)",
        save_path=os.path.join(args.outdir, "supp_representation_pr_auc_curve.png"),
    )
    plot_learning_curves(
        rep_curve_mean, entity_col="representation", metric="balanced_acc",
        title=f"Supplementary: representation comparison under {selected_strategy} (balanced accuracy)",
        save_path=os.path.join(args.outdir, "supp_representation_balanced_acc_curve.png"),
    )

    summary = {
        "metadata": {
            "n_samples_raw": int(len(raw_df)),
            "n_samples_after_label": int(before_dedup),
            "n_samples_after_dedup": int(after_dedup),
            "n_stable": int((df["label"] == 1).sum()),
            "n_unstable": int((df["label"] == 0).sum()),
            "positive_rate": float((df["label"] == 1).mean()),
            "label_mode": args.label_mode,
            "ehull_threshold": args.ehull_threshold,
            "split": args.split,
            "dedup_key": args.dedup_key,
            "test_size": args.test_size,
            "initial_ratio": args.initial_ratio,
            "initial_size": args.initial_size,
            "query_size": args.query_size,
            "n_rounds": args.n_rounds,
            "final_budget_formula": "initial_size + n_rounds * query_size",
            "inner_val_frac": args.inner_val_frac,
            "seeds": args.seeds,
            "train_config": asdict(cfg),
            "duplicate_stats_before_dedup": duplicate_stats_before,
            "dedup_counts": {"before": int(before_dedup), "after": int(after_dedup)},
            "descriptor_filter_counts": {
                "before": int(before_descriptor_filter),
                "after": int(after_descriptor_filter),
            },
            "acquisition_diversity_space": "standardized PiDF descriptor space: wtf, sigma_A, sigma_B, delta_q",
            "selected_strategy": selected_strategy,
            "selected_strategy_mode": args.selected_strategy,
            "strategy_list": strategy_list,
            "representations": list(specs.keys()),
        },
        "feature_sets": specs,
        "strategy_final_budget": strategy_final.to_dict(orient="records"),
        "strategy_aulc_summary": strategy_aulc.to_dict(orient="records"),
        "main_strategy_curve": main_strategy_curve.to_dict(orient="records"),
        "main_strategy_tradeoff_table": strategy_tradeoff_raw.to_dict(orient="records"),
        "representation_final_budget": rep_final.to_dict(orient="records"),
        "representation_aulc_summary": rep_aulc.to_dict(orient="records"),
    }

    with open(os.path.join(args.outdir, "section4_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("===== Section 4 active-learning experiment completed =====")
    print(json.dumps(summary["metadata"], ensure_ascii=False, indent=2))
    print(f"Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()
