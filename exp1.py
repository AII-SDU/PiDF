#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 1 benchmark script for:
Benchmark Performance against Geometric and Compositional Baselines

This script is written on top of the user's original benchmark code and focuses
only on the benchmark section. Relative to the original version, it adds:

1) Stronger and clearer baselines
   - Classical TF Threshold
   - Conventional Composition ML
   - Raw Radius/Valence ML
   - Classical TF + ML
   - Weighted TF + ML
   - PiDF + ML

2) Validation-based threshold tuning for all probabilistic models

3) Optional random / group split support (default: random)

4) Composition-level deduplication enabled by default (main benchmark = 254 unique compositions)

5) Structured outputs for main-table, supplementary-table, calibration, and
   per-run diagnostics

Recommended use for Section 1 of Results and Discussion:
- Use the default composition-level deduplication
- First run with --split random
- Keep the same learner across all ML baselines
- Use the group split in the next section on robustness
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
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =========================
# 基础工具
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
    raise ValueError("group split 需要 host_id / chemsys_query / parent_material_id 之一。")



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



def enrich_composition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 A_elements/A_amounts/B_elements/B_amounts 中派生更强的 compositional baseline 所需列。
    若原始列不存在，则尽量保留现有信息，不报错。
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
                rec["B_host"] = sorted(B_elements)[0] if len(B_elements) > 1 else B_elements[0]
                if len(B_elements) > 1:
                    rec["B_sub"] = sorted(B_elements)[1]
        elif mix_site == "B":
            B_host, B_sub, ratio = choose_host_sub(B_elements, B_amounts)
            rec["B_host"] = B_host
            rec["B_sub"] = B_sub
            rec["mix_ratio"] = ratio if ratio is not None else rec["mix_ratio"]
            if len(A_elements) >= 1:
                rec["A_host"] = sorted(A_elements)[0] if len(A_elements) > 1 else A_elements[0]
                if len(A_elements) > 1:
                    rec["A_sub"] = sorted(A_elements)[1]
        else:
            if len(A_elements) >= 1:
                rec["A_host"] = sorted(A_elements)[0] if len(A_elements) > 1 else A_elements[0]
                if len(A_elements) > 1:
                    rec["A_sub"] = sorted(A_elements)[1]
            if len(B_elements) >= 1:
                rec["B_host"] = sorted(B_elements)[0] if len(B_elements) > 1 else B_elements[0]
                if len(B_elements) > 1:
                    rec["B_sub"] = sorted(B_elements)[1]

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
    Build composition-fraction features from normalized_composition if available.
    Oxygen is omitted because O=3 is fixed after normalization.
    """
    out = df.copy()
    if "normalized_composition" not in out.columns:
        return out

    parsed = []
    elements = set()
    for x in out["normalized_composition"].tolist():
        d = {}
        if isinstance(x, str):
            try:
                d = json.loads(x)
            except Exception:
                try:
                    d = ast.literal_eval(x)
                except Exception:
                    d = {}
        elif isinstance(x, dict):
            d = x
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


def canonicalize_norm_comp(x: Any, ndigits: int = 6) -> str:
    """
    Canonical key for composition-level deduplication.

    The input can be a JSON string, a Python-literal dict string, or a dict.
    Elements are sorted alphabetically and amounts are rounded, so equivalent
    normalized compositions are treated as the same key even if the original
    string formatting differs.
    """
    if isinstance(x, dict):
        d = x
    elif isinstance(x, str):
        s = x.strip()
        if s == "":
            return ""
        try:
            d = json.loads(s)
        except Exception:
            try:
                d = ast.literal_eval(s)
            except Exception:
                return s
    else:
        return str(x)

    if not isinstance(d, dict):
        return str(d)

    items = []
    for k, v in d.items():
        try:
            val = round(float(v), ndigits)
        except Exception:
            val = v
        items.append((str(k), val))
    items = sorted(items, key=lambda t: t[0])
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def maybe_deduplicate_by_composition(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    """
    Deduplicate at the normalized-composition level.

    This is enabled by default for the main 254-composition benchmark. If
    energy_above_hull is available, the lowest-E_hull entry is kept for each
    duplicated normalized composition.
    """
    if not enabled:
        print("[warn] composition dedup is disabled; this may keep repeated normalized compositions.")
        return df.copy()

    out = df.copy()
    key_col = None

    if "normalized_composition" in out.columns:
        out["normalized_composition_key"] = out["normalized_composition"].apply(canonicalize_norm_comp)
        key_col = "normalized_composition_key"
    elif "composition_dict" in out.columns:
        out["normalized_composition_key"] = out["composition_dict"].apply(canonicalize_norm_comp)
        key_col = "normalized_composition_key"
    elif "formula_pretty" in out.columns:
        key_col = "formula_pretty"
    else:
        print("[warn] 未找到 normalized_composition / composition_dict / formula_pretty，跳过 composition dedup。")
        return out

    before = len(out)
    n_dup = int(out.duplicated(subset=[key_col]).sum())
    if "energy_above_hull" in out.columns:
        out = out.sort_values(by="energy_above_hull", ascending=True, kind="mergesort")
    out = out.drop_duplicates(subset=[key_col], keep="first").reset_index(drop=True)
    after = len(out)
    print(f"[info] composition dedup: {before} -> {after} (removed={before-after}, duplicated={n_dup}, key={key_col})")
    return out


def check_required_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")



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
    max_group_trials: int = 50,
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
            f"group split 在 {max_group_trials} 次尝试后仍未获得 train/val/test 都含有两类样本的划分。"
        )

    raise ValueError("split_mode 只能是 random / group")


# =========================
# 阈值法 baseline
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
# 模型
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
    model_type: str = "mlp"  # mlp / logreg / rf
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
# 评价函数
# =========================
def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
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
        pred_bin = (probs[mask] >= 0.5).astype(int)
        acc = np.mean(y_true[mask] == pred_bin)
        conf = np.mean(probs[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)
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
# 特征配置
# =========================
def get_feature_sets(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    # composition-vector features derived from normalized compositions
    frac_num = sorted([c for c in df.columns if c.startswith("frac_")])

    # site-identity composition baseline
    conventional_cat = [c for c in ["mix_site", "A_host", "A_sub", "B_host", "B_sub"] if c in df.columns]
    identity_num = [c for c in ["mix_ratio", "n_A_species", "n_B_species"] if c in df.columns]

    # raw physical quantities before PiDF descriptor aggregation
    raw_cat = [c for c in ["mix_site"] if c in df.columns]
    raw_num = [c for c in ["mix_ratio", "r_A_eff", "r_B_eff", "q_A_eff", "q_B_eff", "r_O"] if c in df.columns]

    # stronger conventional baseline: composition/site identity + composition fractions + raw ionic stats
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
# 单次实验
# =========================
def run_threshold_baseline(df: pd.DataFrame, split_mode: str, seed: int, test_size: float, val_size: float) -> Dict[str, Any]:
    train_idx, val_idx, test_idx = make_split(df, split_mode=split_mode, seed=seed, test_size=test_size, val_size=val_size)

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
    }



def run_ml_baseline(
    df: pd.DataFrame,
    split_mode: str,
    seed: int,
    cat_cols: List[str],
    num_cols: List[str],
    cfg: TrainConfig,
    test_size: float,
    val_size: float,
) -> Dict[str, Any]:
    train_idx, val_idx, test_idx = make_split(df, split_mode=split_mode, seed=seed, test_size=test_size, val_size=val_size)

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
    metrics = compute_metrics(y_true=y_test.astype(int), probs=test_probs, threshold=best_thr)
    metrics["val_threshold_best"] = float(best_thr)
    metrics[f"val_{cfg.threshold_objective}_best"] = float(best_val_score)

    return {
        "metrics": metrics,
        "y_test": y_test.astype(int),
        "probs": test_probs,
        "n_features": int(X_train.shape[1]),
        "feature_columns": {
            "categorical": cat_cols,
            "numeric": num_cols,
        },
    }


# =========================
# 汇总与绘图
# =========================
def summarize_metric_dicts(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metric_names = list(results[0]["metrics"].keys())
    out: Dict[str, Dict[str, float]] = {}
    for metric in metric_names:
        vals = np.array([r["metrics"][metric] for r in results], dtype=float)
        out[metric] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
        }
    if "n_features" in results[0]:
        vals = np.array([r["n_features"] for r in results], dtype=float)
        out["n_features"] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return out



def to_wide_table(summary: Dict[str, Any], methods: List[str], metric_order: List[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        row = {"method": method}
        for metric in metric_order:
            if metric in summary[method]:
                row[f"{metric}_mean"] = summary[method][metric]["mean"]
                row[f"{metric}_std"] = summary[method][metric]["std"]
        rows.append(row)
    return pd.DataFrame(rows)



def plot_roc_curves(curve_data: Dict[str, Tuple[np.ndarray, np.ndarray]], save_path: str) -> None:
    plt.figure(figsize=(6, 5))
    for name, (y_true, probs) in curve_data.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves (seed 0 split)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def plot_pr_curves(curve_data: Dict[str, Tuple[np.ndarray, np.ndarray]], save_path: str) -> None:
    plt.figure(figsize=(6, 5))
    for name, (y_true, probs) in curve_data.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curves (seed 0 split)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================
# 主程序
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./mp_data/exp42_perovskite_like_subset.csv")
    parser.add_argument("--outdir", type=str, default="runs/section1_benchmark")
    parser.add_argument("--split", type=str, default="random", choices=["random", "group"])
    parser.add_argument("--label-mode", type=str, default="ehull", choices=["label", "is_stable", "ehull"])
    parser.add_argument("--ehull-threshold", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--model-type", type=str, default="logreg", choices=["mlp", "logreg", "rf"])
    parser.add_argument("--threshold-objective", type=str, default="balanced_acc", choices=["balanced_acc", "f1"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dedup-composition", dest="dedup_composition", action="store_true", default=True, help="Enable composition-level deduplication (default: enabled).")
    parser.add_argument("--no-dedup-composition", dest="dedup_composition", action="store_false", help="Disable composition-level deduplication.")
    parser.add_argument("--save-clean-dataset", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    set_seed(0)

    df = pd.read_csv(args.csv)
    df = build_label_column(df, args.label_mode, args.ehull_threshold)
    df = enrich_composition_columns(df)
    df = enrich_fraction_features(df)
    df = maybe_deduplicate_by_composition(df, enabled=args.dedup_composition)

    check_required_columns(df, ["classical_t", "wtf", "sigma_A", "sigma_B", "delta_q"])
    df = df[df[["classical_t", "wtf", "sigma_A", "sigma_B", "delta_q"]].notna().all(axis=1)].copy()
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("清洗后没有可用样本。")

    # 检查 composition duplicates 但不强制删除
    duplicate_stats = {}
    if "normalized_composition" in df.columns:
        duplicate_stats["n_duplicate_normalized_composition"] = int(df.duplicated(subset=["normalized_composition"]).sum())
    if "formula_pretty" in df.columns:
        duplicate_stats["n_duplicate_formula_pretty"] = int(df.duplicated(subset=["formula_pretty"]).sum())

    if args.save_clean_dataset:
        df.to_csv(os.path.join(args.outdir, "clean_dataset.csv"), index=False)

    feature_sets = get_feature_sets(df)
    cfg = TrainConfig(
        model_type=args.model_type,
        threshold_objective=args.threshold_objective,
        device=args.device,
    )

    method_order = [
        "Classical TF Threshold",
        "Composition Vector + ML",
        "Conventional Composition/Site + ML",
        "Raw Radius/Valence + ML",
        "Classical TF + ML",
        "Weighted TF + ML",
        "PiDF + ML",
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in method_order}
    curve_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    per_run_rows: List[Dict[str, Any]] = []

    for i, seed in enumerate(args.seeds):
        set_seed(seed)

        # threshold baseline
        thr_res = run_threshold_baseline(df, split_mode=args.split, seed=seed, test_size=args.test_size, val_size=args.val_size)
        all_results["Classical TF Threshold"].append(thr_res)
        for metric_name, metric_val in thr_res["metrics"].items():
            per_run_rows.append({
                "seed": seed,
                "method": "Classical TF Threshold",
                "metric": metric_name,
                "value": metric_val,
            })

        # ML baselines
        for method_name in method_order[1:]:
            spec = feature_sets[method_name]
            res = run_ml_baseline(
                df=df,
                split_mode=args.split,
                seed=seed,
                cat_cols=spec["categorical"],
                num_cols=spec["numeric"],
                cfg=cfg,
                test_size=args.test_size,
                val_size=args.val_size,
            )
            all_results[method_name].append(res)
            for metric_name, metric_val in res["metrics"].items():
                per_run_rows.append({
                    "seed": seed,
                    "method": method_name,
                    "metric": metric_name,
                    "value": metric_val,
                })
            per_run_rows.append({
                "seed": seed,
                "method": method_name,
                "metric": "n_features",
                "value": res["n_features"],
            })

            if i == 0:
                curve_results[method_name] = (res["y_test"], res["probs"])

    summary: Dict[str, Any] = {
        "n_samples": int(len(df)),
        "n_stable": int((df["label"] == 1).sum()),
        "n_unstable": int((df["label"] == 0).sum()),
        "positive_rate": float(df["label"].mean()),
        "split_mode": args.split,
        "label_mode": args.label_mode,
        "ehull_threshold": args.ehull_threshold,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "seeds": args.seeds,
        "train_config": asdict(cfg),
        "duplicate_stats": duplicate_stats,
        "feature_sets": feature_sets,
    }

    for method_name, results in all_results.items():
        summary[method_name] = summarize_metric_dicts(results)

    # 保存 JSON 总结
    with open(os.path.join(args.outdir, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # per-run 明细
    pd.DataFrame(per_run_rows).to_csv(os.path.join(args.outdir, "benchmark_per_run_metrics.csv"), index=False)

    # 主表：正文常用
    main_metrics = ["roc_auc", "pr_auc", "f1", "balanced_acc", "precision", "recall"]
    main_table = to_wide_table(summary, method_order, main_metrics)
    main_table.to_csv(os.path.join(args.outdir, "benchmark_main_table.csv"), index=False)

    # 补充表：含 calibration / threshold
    supp_metrics = [
        "roc_auc", "pr_auc", "f1", "balanced_acc", "precision", "recall",
        "brier", "ece", "val_threshold_best", "val_balanced_acc_best", "val_f1_best",
        "threshold_low", "threshold_high", "n_features",
    ]
    supp_table = to_wide_table(summary, method_order, supp_metrics)
    supp_table.to_csv(os.path.join(args.outdir, "benchmark_supplementary_table.csv"), index=False)

    # calibration 单独导出（更清楚）
    calib_rows = []
    for method_name in method_order:
        method_summary = summary[method_name]
        row = {"method": method_name}
        for metric in ["brier", "ece"]:
            if metric in method_summary:
                row[f"{metric}_mean"] = method_summary[metric]["mean"]
                row[f"{metric}_std"] = method_summary[metric]["std"]
        calib_rows.append(row)
    pd.DataFrame(calib_rows).to_csv(os.path.join(args.outdir, "benchmark_calibration.csv"), index=False)

    # 绘图（只对概率模型）
    if len(curve_results) > 0:
        plot_roc_curves(curve_results, os.path.join(args.outdir, "benchmark_roc_curves_seed0.png"))
        plot_pr_curves(curve_results, os.path.join(args.outdir, "benchmark_pr_curves_seed0.png"))

    print("===== Section 1 benchmark completed =====")
    print(json.dumps({
        "n_samples": summary["n_samples"],
        "n_stable": summary["n_stable"],
        "n_unstable": summary["n_unstable"],
        "split_mode": summary["split_mode"],
        "model_type": cfg.model_type,
        "outdir": args.outdir,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
