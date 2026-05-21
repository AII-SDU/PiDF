#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 3 script for:
What Drives PiDF? Ablation and Descriptor-Level Interpretation

This script is built on top of the user's original benchmark code and the
Section-1/Section-2 logic, but focuses on descriptor ablation and
interpretability analysis.

It provides:

1) Descriptor-only ablation
   - wtf only
   - wtf sigma
   - wtf deltaq
   - PiDF full

2) Incremental ablation on top of a stronger conventional composition/site
   baseline
   - Conventional base
   - Base + wtf
   - Base + wtf sigma
   - Base + wtf deltaq
   - Base + PiDF full

3) Descriptor-level subgroup analysis
   - low/high disorder subsets using train-set median of sigma_total
   - charge-balanced vs charge-mismatched subsets using delta_q

4) Error analysis for PiDF full
   - TP / TN / FP / FN descriptor statistics on the test set

5) Optional coefficient extraction for logistic regression
   - descriptor-only models: all coefficients
   - incremental models: PiDF-related coefficients only

Recommended use for Section 3:
- keep the same learner as previous sections (e.g. --model-type logreg)
- enable deduplication (recommended: --dedup-key auto)
- use random split for the main ablation analysis
- use the subgroup tables to support the discussion on sigma and delta_q
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

    The normalized_composition column may contain floating-point values such as
    0.30000000000000004 and 0.3 for the same nominal composition. Rounding the
    amounts before serialization makes deduplication more robust and consistent
    with the Section-1 and Section-2 scripts.
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


def add_descriptor_helpers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sigma_A" in out.columns and "sigma_B" in out.columns:
        out["sigma_total"] = out["sigma_A"].astype(float) + out["sigma_B"].astype(float)
    elif "sigma_total" not in out.columns:
        out["sigma_total"] = np.nan
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


def make_split(
    df: pd.DataFrame,
    split_mode: str,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    max_group_trials: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(df))
    y = df["label"].values

    if split_mode == "random":
        trainval_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
        y_trainval = y[trainval_idx]
        val_ratio = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_ratio,
            random_state=seed,
            stratify=y_trainval,
        )
        return train_idx, val_idx, test_idx

    if split_mode == "group":
        group_col = choose_group_column(df)
        groups = df[group_col].values
        val_ratio = val_size / (1.0 - test_size)

        for trial in range(max_group_trials):
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
            f"group split failed to produce train/val/test containing both classes after {max_group_trials} trials"
        )

    raise ValueError("split_mode must be one of: random / group")


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

    raise ValueError("model_type must be one of: mlp / logreg / rf")



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
    probs: Optional[np.ndarray] = None,
    pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    compute_auc: bool = True,
    include_calibration: bool = True,
) -> Dict[str, float]:
    if pred is None:
        if probs is None:
            raise ValueError("compute_metrics needs probs or pred")
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
def _ordered_unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))



def get_conventional_base_spec(df: pd.DataFrame) -> Dict[str, List[str]]:
    frac_num = sorted([c for c in df.columns if c.startswith("frac_")])
    conventional_cat = [c for c in ["mix_site", "A_host", "A_sub", "B_host", "B_sub"] if c in df.columns]
    identity_num = [c for c in ["mix_ratio", "n_A_species", "n_B_species"] if c in df.columns]
    raw_num = [c for c in ["r_A_eff", "r_B_eff", "q_A_eff", "q_B_eff", "r_O"] if c in df.columns]
    conventional_num = _ordered_unique(identity_num + frac_num + raw_num)
    return {
        "categorical": conventional_cat,
        "numeric": conventional_num,
    }



def get_descriptor_only_ablation_sets(df: pd.DataFrame, include_mix_site: bool = False) -> Dict[str, Dict[str, List[str]]]:
    cat = ["mix_site"] if include_mix_site and "mix_site" in df.columns else []
    sets = {
        "wtf only": {
            "categorical": cat,
            "numeric": [c for c in ["wtf"] if c in df.columns],
        },
        "wtf sigma": {
            "categorical": cat,
            "numeric": [c for c in ["wtf", "sigma_A", "sigma_B"] if c in df.columns],
        },
        "wtf deltaq": {
            "categorical": cat,
            "numeric": [c for c in ["wtf", "delta_q"] if c in df.columns],
        },
        "PiDF full": {
            "categorical": cat,
            "numeric": [c for c in ["wtf", "sigma_A", "sigma_B", "delta_q"] if c in df.columns],
        },
    }
    return sets



def get_incremental_ablation_sets(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    base = get_conventional_base_spec(df)
    base_cat = list(base["categorical"])
    base_num = list(base["numeric"])
    sets = {
        "Conventional base": {
            "categorical": base_cat,
            "numeric": base_num,
        },
        "Base + wtf": {
            "categorical": base_cat,
            "numeric": _ordered_unique(base_num + [c for c in ["wtf"] if c in df.columns]),
        },
        "Base + wtf sigma": {
            "categorical": base_cat,
            "numeric": _ordered_unique(base_num + [c for c in ["wtf", "sigma_A", "sigma_B"] if c in df.columns]),
        },
        "Base + wtf deltaq": {
            "categorical": base_cat,
            "numeric": _ordered_unique(base_num + [c for c in ["wtf", "delta_q"] if c in df.columns]),
        },
        "Base + PiDF full": {
            "categorical": base_cat,
            "numeric": _ordered_unique(base_num + [c for c in ["wtf", "sigma_A", "sigma_B", "delta_q"] if c in df.columns]),
        },
    }
    return sets



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



def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if name == "cat":
                try:
                    enc = trans.named_steps["onehot"]
                    enc_names = list(enc.get_feature_names_out(cols))
                    names.extend([f"cat__{x}" for x in enc_names])
                except Exception:
                    names.extend([f"cat__{c}" for c in cols])
            elif name == "num":
                names.extend([f"num__{c}" for c in cols])
        return names


# =========================
# Single-run execution
# =========================
def run_ml_method(
    df: pd.DataFrame,
    split_mode: str,
    seed: int,
    cat_cols: List[str],
    num_cols: List[str],
    cfg: TrainConfig,
    test_size: float,
    val_size: float,
    return_details: bool = False,
) -> Dict[str, Any]:
    train_idx, val_idx, test_idx = make_split(df, split_mode=split_mode, seed=seed, test_size=test_size, val_size=val_size)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    used_cols = list(cat_cols) + list(num_cols)
    preprocessor = build_preprocessor(cat_cols, num_cols)

    X_train = preprocessor.fit_transform(train_df[used_cols])
    X_val = preprocessor.transform(val_df[used_cols])
    X_test = preprocessor.transform(test_df[used_cols])

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()

    feature_names = get_feature_names(preprocessor)

    y_train = train_df["label"].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    model, val_probs, test_probs = fit_predict_model(X_train, y_train, X_val, y_val, X_test, cfg)
    best_thr, best_val_score = tune_probability_threshold(y_val.astype(int), val_probs, cfg.threshold_objective)
    metrics = compute_metrics(y_true=y_test.astype(int), probs=test_probs, threshold=best_thr)
    metrics["val_threshold_best"] = float(best_thr)
    metrics[f"val_{cfg.threshold_objective}_best"] = float(best_val_score)
    metrics["n_features"] = float(X_train.shape[1])

    result: Dict[str, Any] = {
        "metrics": metrics,
        "feature_names": feature_names,
    }
    if return_details:
        result.update({
            "model": model,
            "preprocessor": preprocessor,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "y_test": y_test.astype(int),
            "test_probs": test_probs,
            "test_pred": (test_probs >= best_thr).astype(int),
            "threshold": float(best_thr),
        })
    return result


# =========================
# Summaries / tables
# =========================
def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metric_names = list(results[0]["metrics"].keys())
    out: Dict[str, Dict[str, float]] = {}
    for metric in metric_names:
        vals = np.array([r["metrics"][metric] for r in results], dtype=float)
        out[metric] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
        }
    return out



def summary_to_rows(summary_block: Dict[str, Dict[str, Dict[str, float]]], method_order: List[str], main_metrics: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    main_rows = []
    supp_rows = []
    for method in method_order:
        for metric, stats in summary_block[method].items():
            row = {
                "method": method,
                "metric": metric,
                "mean": stats["mean"],
                "std": stats["std"],
            }
            supp_rows.append(row)
            if metric in main_metrics:
                main_rows.append(row)
    return pd.DataFrame(main_rows), pd.DataFrame(supp_rows)


# =========================
# Subgroup analysis
# =========================
def compute_subset_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float, mask: np.ndarray) -> Dict[str, float]:
    if mask.sum() == 0:
        return {
            "n_samples": 0.0,
            "n_positive": 0.0,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "f1": np.nan,
            "balanced_acc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
        }

    ys = y_true[mask]
    ps = probs[mask]
    pred = (ps >= threshold).astype(int)
    out = {
        "n_samples": float(len(ys)),
        "n_positive": float(np.sum(ys == 1)),
    }
    compute_auc = len(np.unique(ys)) > 1
    out.update(compute_metrics(ys, probs=ps, pred=pred, threshold=threshold, compute_auc=compute_auc, include_calibration=False))
    return out



def summarize_long_table(df: pd.DataFrame, value_cols: List[str], group_cols: List[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {k: v for k, v in zip(group_cols, keys)}
        for metric in value_cols:
            vals = pd.to_numeric(sub[metric], errors="coerce").values.astype(float)
            rows.append({
                **base,
                "metric": metric,
                "mean": float(np.nanmean(vals)) if len(vals) > 0 else np.nan,
                "std": float(np.nanstd(vals)) if len(vals) > 0 else np.nan,
            })
    return pd.DataFrame(rows)



def make_subgroup_masks(train_df: pd.DataFrame, test_df: pd.DataFrame, charge_tol: float = 1e-8) -> Dict[str, np.ndarray]:
    sigma_thr = float(np.nanmedian(train_df["sigma_total"].astype(float).values))
    test_sigma = test_df["sigma_total"].astype(float).values
    test_delta = test_df["delta_q"].astype(float).values
    return {
        "all": np.ones(len(test_df), dtype=bool),
        "low_disorder": test_sigma < sigma_thr,
        "high_disorder": test_sigma >= sigma_thr,
        "charge_balanced": test_delta <= charge_tol,
        "charge_mismatched": test_delta > charge_tol,
    }



def collect_subgroup_rows(
    family_name: str,
    method_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    charge_tol: float = 1e-8,
) -> List[Dict[str, Any]]:
    masks = make_subgroup_masks(train_df, test_df, charge_tol=charge_tol)
    rows = []
    for subset_name, mask in masks.items():
        stats = compute_subset_metrics(y_test, probs, threshold, mask)
        row = {
            "family": family_name,
            "method": method_name,
            "subset": subset_name,
        }
        row.update(stats)
        rows.append(row)
    return rows



def build_gain_rows(subgroup_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("descriptor_only", "wtf sigma", "wtf only", "gain_sigma_vs_wtf"),
        ("descriptor_only", "wtf deltaq", "wtf only", "gain_deltaq_vs_wtf"),
        ("descriptor_only", "PiDF full", "wtf only", "gain_full_vs_wtf"),
        ("incremental", "Base + wtf sigma", "Conventional base", "gain_sigma_vs_base"),
        ("incremental", "Base + wtf deltaq", "Conventional base", "gain_deltaq_vs_base"),
        ("incremental", "Base + PiDF full", "Conventional base", "gain_full_vs_base"),
    ]
    metrics = ["roc_auc", "pr_auc", "f1", "balanced_acc", "precision", "recall"]
    rows = []
    run_group_cols = ["run_id", "family", "subset", "method"]
    if not set(run_group_cols).issubset(subgroup_df.columns):
        return pd.DataFrame()

    piv = subgroup_df.pivot_table(index=["run_id", "family", "subset"], columns="method", values=metrics)
    for family, num_method, den_method, comp_name in comparisons:
        if (metrics[0], num_method) not in piv.columns or (metrics[0], den_method) not in piv.columns:
            continue
        for idx in piv.index:
            run_id, fam, subset = idx
            if fam != family:
                continue
            row = {
                "run_id": run_id,
                "family": fam,
                "subset": subset,
                "comparison": comp_name,
            }
            valid = False
            for metric in metrics:
                try:
                    num = piv.loc[idx, (metric, num_method)]
                    den = piv.loc[idx, (metric, den_method)]
                    row[metric] = float(num - den)
                    valid = True
                except Exception:
                    row[metric] = np.nan
            if valid:
                rows.append(row)
    return pd.DataFrame(rows)


# =========================
# Error analysis
# =========================
def collect_error_rows(
    family_name: str,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    pred: np.ndarray,
) -> List[Dict[str, Any]]:
    labels = {
        "TP": (y_true == 1) & (pred == 1),
        "TN": (y_true == 0) & (pred == 0),
        "FP": (y_true == 0) & (pred == 1),
        "FN": (y_true == 1) & (pred == 0),
    }
    descriptor_cols = [c for c in ["wtf", "sigma_A", "sigma_B", "sigma_total", "delta_q"] if c in test_df.columns]
    rows = []
    for err_name, mask in labels.items():
        sub = test_df.loc[mask].copy()
        row: Dict[str, Any] = {
            "family": family_name,
            "error_type": err_name,
            "n_samples": int(mask.sum()),
            "prob_mean": float(np.nanmean(probs[mask])) if mask.sum() > 0 else np.nan,
        }
        for col in descriptor_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").values.astype(float) if len(sub) > 0 else np.array([], dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
            row[f"{col}_std"] = float(np.nanstd(vals)) if len(vals) > 0 else np.nan
        rows.append(row)
    return rows


# =========================
# Coefficient extraction
# =========================
def collect_logreg_coefficients(
    family_name: str,
    method_name: str,
    model: Any,
    feature_names: List[str],
    only_pidf_related: bool = False,
) -> List[Dict[str, Any]]:
    if not isinstance(model, LogisticRegression):
        return []
    coef = np.ravel(model.coef_)
    rows = []
    keep_terms = {"wtf", "sigma_A", "sigma_B", "delta_q", "classical_t"}
    for feat, val in zip(feature_names, coef):
        base_name = feat.split("__")[-1]
        if only_pidf_related and base_name not in keep_terms:
            continue
        rows.append({
            "family": family_name,
            "method": method_name,
            "feature": feat,
            "coefficient": float(val),
        })
    return rows


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./mp_data/exp42_perovskite_like_subset.csv")
    parser.add_argument("--outdir", type=str, default="runs/section3_ablation")
    parser.add_argument("--split", type=str, default="random", choices=["random", "group"])
    parser.add_argument("--dedup-key", type=str, default="auto", choices=["none", "auto", "normalized_composition_canonical", "formula_pretty"])
    parser.add_argument("--label-mode", type=str, default="ehull", choices=["label", "is_stable", "ehull"])
    parser.add_argument("--ehull-threshold", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--model-type", type=str, default="logreg", choices=["logreg", "mlp", "rf"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--threshold-objective", type=str, default="balanced_acc", choices=["balanced_acc", "f1"])
    parser.add_argument("--descriptor-only-include-mix-site", action="store_true")
    parser.add_argument("--families", nargs="+", default=["descriptor_only", "incremental"], choices=["descriptor_only", "incremental"])
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--charge-tol", type=float, default=1e-8)
    parser.add_argument("--save-clean-dataset", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    set_seed(0)

    raw_df = pd.read_csv(args.csv)
    df = build_label_column(raw_df, args.label_mode, args.ehull_threshold)
    n_samples_after_label = len(df)

    df = enrich_composition_columns(df)
    df = enrich_fraction_features(df)
    df = canonicalize_normalized_composition(df)
    df = add_descriptor_helpers(df)

    duplicate_stats_before = {
        "n_duplicate_normalized_composition": int(df["normalized_composition_canonical"].duplicated().sum()) if "normalized_composition_canonical" in df.columns else None,
        "n_duplicate_formula_pretty": int(df["formula_pretty"].duplicated().sum()) if "formula_pretty" in df.columns else None,
    }

    before_dedup = len(df)
    df = deduplicate_dataset(df, args.dedup_key)
    after_dedup = len(df)

    required_cols = ["wtf", "sigma_A", "sigma_B", "delta_q"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Required PiDF columns are missing: {missing_required}")

    before_descriptor_filter = len(df)
    df = df[df[required_cols].notna().all(axis=1)].copy().reset_index(drop=True)
    after_descriptor_filter = len(df)

    if len(df) == 0:
        raise ValueError("No usable samples remain after descriptor filtering.")

    # Recompute helper descriptors after deduplication/filtering to keep all
    # downstream subgroup and error-analysis tables aligned with the final data.
    df = add_descriptor_helpers(df)

    cfg = TrainConfig(
        model_type=args.model_type,
        device=args.device,
        threshold_objective=args.threshold_objective,
        logreg_c=args.logreg_c,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
    )

    if args.save_clean_dataset:
        df.to_csv(os.path.join(args.outdir, "clean_dataset.csv"), index=False)

    descriptor_sets = get_descriptor_only_ablation_sets(df, include_mix_site=args.descriptor_only_include_mix_site)
    incremental_sets = get_incremental_ablation_sets(df)

    family_feature_sets: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    if "descriptor_only" in args.families:
        family_feature_sets["descriptor_only"] = descriptor_sets
    if "incremental" in args.families:
        family_feature_sets["incremental"] = incremental_sets

    family_run_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        fam: {method: [] for method in feature_sets.keys()} for fam, feature_sets in family_feature_sets.items()
    }
    subgroup_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    coefficient_rows: List[Dict[str, Any]] = []
    per_run_rows: List[Dict[str, Any]] = []

    for run_idx, seed in enumerate(args.seeds):
        set_seed(seed)
        for family_name, feature_sets in family_feature_sets.items():
            for method_name, spec in feature_sets.items():
                res = run_ml_method(
                    df=df,
                    split_mode=args.split,
                    seed=seed,
                    cat_cols=spec["categorical"],
                    num_cols=spec["numeric"],
                    cfg=cfg,
                    test_size=args.test_size,
                    val_size=args.val_size,
                    return_details=True,
                )
                family_run_results[family_name][method_name].append(res)

                # per-run metrics table
                metric_row = {"run_id": run_idx, "seed": seed, "family": family_name, "method": method_name}
                metric_row.update(res["metrics"])
                per_run_rows.append(metric_row)

                # subgroup analysis
                subgroup_part = collect_subgroup_rows(
                    family_name=family_name,
                    method_name=method_name,
                    train_df=res["train_df"],
                    test_df=res["test_df"],
                    y_test=res["y_test"],
                    probs=res["test_probs"],
                    threshold=res["threshold"],
                    charge_tol=args.charge_tol,
                )
                for row in subgroup_part:
                    row["run_id"] = run_idx
                    row["seed"] = seed
                subgroup_rows.extend(subgroup_part)

                # error analysis for the full model of each family
                if (family_name == "descriptor_only" and method_name == "PiDF full") or (
                    family_name == "incremental" and method_name == "Base + PiDF full"
                ):
                    err_part = collect_error_rows(
                        family_name=family_name,
                        test_df=res["test_df"],
                        y_true=res["y_test"],
                        probs=res["test_probs"],
                        pred=res["test_pred"],
                    )
                    for row in err_part:
                        row["run_id"] = run_idx
                        row["seed"] = seed
                    error_rows.extend(err_part)

                # coefficient extraction for logreg
                if cfg.model_type == "logreg":
                    coeff_part = collect_logreg_coefficients(
                        family_name=family_name,
                        method_name=method_name,
                        model=res["model"],
                        feature_names=res["feature_names"],
                        only_pidf_related=(family_name == "incremental"),
                    )
                    for row in coeff_part:
                        row["run_id"] = run_idx
                        row["seed"] = seed
                    coefficient_rows.extend(coeff_part)

    # Summaries by family
    overall_summary: Dict[str, Any] = {
        "metadata": {
            "n_samples_raw": int(len(raw_df)),
            "n_samples_after_label": int(n_samples_after_label),
            "n_samples_after_dedup": int(len(df)),
            "n_stable": int((df["label"] == 1).sum()),
            "n_unstable": int((df["label"] == 0).sum()),
            "positive_rate": float((df["label"] == 1).mean()),
            "label_mode": args.label_mode,
            "ehull_threshold": float(args.ehull_threshold),
            "split": args.split,
            "dedup_key": args.dedup_key,
            "test_size": float(args.test_size),
            "val_size": float(args.val_size),
            "seeds": args.seeds,
            "train_config": asdict(cfg),
            "duplicate_stats_before_dedup": duplicate_stats_before,
            "dedup_counts": {"before": int(before_dedup), "after": int(after_dedup)},
            "descriptor_filter_counts": {
                "before": int(before_descriptor_filter),
                "after": int(after_descriptor_filter),
            },
            "families": args.families,
            "descriptor_only_include_mix_site": bool(args.descriptor_only_include_mix_site),
            "charge_tol": float(args.charge_tol),
        },
        "feature_sets": family_feature_sets,
        "results": {},
    }

    for family_name, method_results in family_run_results.items():
        overall_summary["results"][family_name] = {}
        for method_name, runs in method_results.items():
            overall_summary["results"][family_name][method_name] = summarize_results(runs)

    with open(os.path.join(args.outdir, "section3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    # Main / supplementary tables per family
    main_metrics = ["roc_auc", "pr_auc", "f1", "balanced_acc"]
    for family_name, method_results in overall_summary["results"].items():
        method_order = list(method_results.keys())
        main_df, supp_df = summary_to_rows(method_results, method_order, main_metrics)
        main_df.to_csv(os.path.join(args.outdir, f"{family_name}_main_table.csv"), index=False)
        supp_df.to_csv(os.path.join(args.outdir, f"{family_name}_supplementary_table.csv"), index=False)

        # Calibration-specific rows
        calib_rows = []
        for method in method_order:
            for metric in ["brier", "ece", "val_threshold_best", f"val_{cfg.threshold_objective}_best", "n_features"]:
                if metric in method_results[method]:
                    calib_rows.append({
                        "method": method,
                        "metric": metric,
                        "mean": method_results[method][metric]["mean"],
                        "std": method_results[method][metric]["std"],
                    })
        pd.DataFrame(calib_rows).to_csv(os.path.join(args.outdir, f"{family_name}_calibration_table.csv"), index=False)

    # Per-run table
    per_run_df = pd.DataFrame(per_run_rows)
    per_run_df.to_csv(os.path.join(args.outdir, "section3_per_run_metrics.csv"), index=False)

    # Subgroup analysis tables
    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(os.path.join(args.outdir, "section3_subgroup_per_run.csv"), index=False)
    if len(subgroup_df) > 0:
        subgroup_summary_df = summarize_long_table(
            subgroup_df,
            value_cols=["n_samples", "n_positive", "roc_auc", "pr_auc", "f1", "balanced_acc", "precision", "recall"],
            group_cols=["family", "method", "subset"],
        )
        subgroup_summary_df.to_csv(os.path.join(args.outdir, "section3_subgroup_summary.csv"), index=False)

        gain_df = build_gain_rows(subgroup_df)
        gain_df.to_csv(os.path.join(args.outdir, "section3_subgroup_gains_per_run.csv"), index=False)
        if len(gain_df) > 0:
            gain_summary_df = summarize_long_table(
                gain_df,
                value_cols=["roc_auc", "pr_auc", "f1", "balanced_acc", "precision", "recall"],
                group_cols=["family", "comparison", "subset"],
            )
            gain_summary_df.to_csv(os.path.join(args.outdir, "section3_subgroup_gain_summary.csv"), index=False)

    # Error analysis tables
    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(os.path.join(args.outdir, "section3_error_analysis_per_run.csv"), index=False)
    if len(error_df) > 0:
        value_cols = [c for c in error_df.columns if c not in ["run_id", "seed", "family", "error_type"]]
        error_summary_df = summarize_long_table(error_df, value_cols=value_cols, group_cols=["family", "error_type"])
        error_summary_df.to_csv(os.path.join(args.outdir, "section3_error_analysis_summary.csv"), index=False)

    # Coefficient tables
    coeff_df = pd.DataFrame(coefficient_rows)
    coeff_df.to_csv(os.path.join(args.outdir, "section3_coefficients_per_run.csv"), index=False)
    if len(coeff_df) > 0:
        coeff_summary_df = summarize_long_table(
            coeff_df,
            value_cols=["coefficient"],
            group_cols=["family", "method", "feature"],
        )
        coeff_summary_df.to_csv(os.path.join(args.outdir, "section3_coefficients_summary.csv"), index=False)

    print("===== Section 3 (ablation + interpretation) finished =====")
    print(json.dumps(overall_summary["metadata"], ensure_ascii=False, indent=2))
    print(f"Results directory: {args.outdir}")


if __name__ == "__main__":
    main()
