#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据单组分 ABO3 宿主数据和 ionic_radii.csv 生成多组分候选，并计算：
- classical_t
- wTf
- sigma_A
- sigma_B
- delta_q

输入：
1) host_csv: 至少包含以下列（推荐使用前一步生成的 mp_abo3_with_t.csv）
   - material_id
   - formula_pretty
   - assigned_A_element
   - assigned_B_element
   - A_oxidation_state
   - B_oxidation_state
   - r_A
   - r_B
   - r_O
   其中也可以缺少部分半径列，脚本会尝试从 ionic_radii.csv 重算。

2) ionic_radii.csv: 包含
   - element
   - oxidation_state
   - coordination
   - ionic_radius
   - site_group   (建议有，A/B/X)

输出：
- multicomponent_candidates.csv

候选形式：
- A-site mixing: A_(1-x) A'_x B O3
- B-site mixing: A B_(1-y) B'_y O3

默认只做二元混合；每次只混一个晶位。
"""

import os
import math
import ast
import json
import argparse
from typing import Dict, Tuple, Optional, List

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-csv", type=str, default="./mp_data/mp_abo3_with_t.csv", help="单组分宿主 CSV 路径")
    parser.add_argument("--radii-csv", type=str, default="./mp_data/ionic_radii.csv", help="ionic_radii.csv 路径")
    parser.add_argument("--output-csv", type=str, default="./mp_data/multicomponent_candidates.csv", help="输出路径")
    parser.add_argument(
        "--mix-ratios",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5",
        help="混合比例，逗号分隔，例如 0.1,0.2,0.3,0.4,0.5",
    )
    parser.add_argument(
        "--mix-mode",
        type=str,
        default="both",
        choices=["A", "B", "both"],
        help="生成 A 位混合、B 位混合或两者都生成",
    )
    parser.add_argument(
        "--exclude-host-element",
        action="store_true",
        help="若开启，则替代元素集合中排除与宿主相同的元素",
    )
    parser.add_argument(
        "--enforce-charge-neutrality",
        action="store_true",
        help="若开启，则只保留 delta_q=0 的候选",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=-1,
        help="最大输出候选数，-1 表示不限制（调试时可用）",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_radii_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"element", "oxidation_state", "coordination", "ionic_radius"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"radii_csv 缺少必要列: {missing}")
    df["element"] = df["element"].astype(str)
    df["oxidation_state"] = df["oxidation_state"].astype(int)
    df["coordination"] = df["coordination"].astype(int)
    df["ionic_radius"] = df["ionic_radius"].astype(float)
    if "site_group" in df.columns:
        df["site_group"] = df["site_group"].astype(str)
    else:
        df["site_group"] = None
    return df


def radius_lookup_from_df(radii_df: pd.DataFrame) -> Dict[Tuple[str, int, int], float]:
    lookup: Dict[Tuple[str, int, int], float] = {}
    for _, row in radii_df.iterrows():
        lookup[(row["element"], int(row["oxidation_state"]), int(row["coordination"]))] = float(row["ionic_radius"])
    return lookup


def get_radius(lookup: Dict[Tuple[str, int, int], float], element: str, ox: int, cn: int) -> Optional[float]:
    return lookup.get((str(element), int(ox), int(cn)), None)


def parse_mix_ratios(s: str) -> List[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if not (0.0 < v < 1.0):
            raise ValueError(f"混合比例必须在 (0,1) 内，当前为 {v}")
        vals.append(v)
    if not vals:
        raise ValueError("至少需要一个混合比例")
    return vals


def build_substitution_sets(radii_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 默认 A: CN=12, B: CN=6, X: O2-, CN=6
    if "site_group" in radii_df.columns and radii_df["site_group"].notna().any():
        a_df = radii_df[radii_df["site_group"] == "A"].copy()
        b_df = radii_df[radii_df["site_group"] == "B"].copy()
        x_df = radii_df[radii_df["site_group"] == "X"].copy()
    else:
        a_df = radii_df[radii_df["coordination"] == 12].copy()
        b_df = radii_df[radii_df["coordination"] == 6].copy()
        x_df = radii_df[(radii_df["element"] == "O") & (radii_df["oxidation_state"] == -2) & (radii_df["coordination"] == 6)].copy()
    return a_df, b_df, x_df


def safe_get_host_radius(row: pd.Series, field: str, fallback_element: str, fallback_ox: int, cn: int, lookup) -> float:
    val = row.get(field, None)
    if pd.notna(val):
        return float(val)
    r = get_radius(lookup, fallback_element, fallback_ox, cn)
    if r is None:
        raise ValueError(f"无法获得 {fallback_element}^{fallback_ox} CN={cn} 的半径")
    return float(r)


def format_formula_A_mix(A: str, A_sub: str, x: float, B: str) -> str:
    return f"{A}_(1-{x:g}){A_sub}_{x:g}{B}O_3"


def format_formula_B_mix(A: str, B: str, B_sub: str, y: float) -> str:
    return f"{A}{B}_(1-{y:g}){B_sub}_{y:g}O_3"


def classical_t(r_A: float, r_B: float, r_O: float) -> float:
    return (r_A + r_O) / (math.sqrt(2.0) * (r_B + r_O))


def weighted_tf(r_A_eff: float, r_B_eff: float, r_O: float) -> float:
    return (r_A_eff + r_O) / (math.sqrt(2.0) * (r_B_eff + r_O))


def sigma_binary(x: float, r_host: float, r_sub: float) -> float:
    r_eff = (1.0 - x) * r_host + x * r_sub
    return math.sqrt((1.0 - x) * (r_host - r_eff) ** 2 + x * (r_sub - r_eff) ** 2)


def delta_q_binary(q_A_eff: float, q_B_eff: float, q_O: float = -2.0) -> float:
    return abs(q_A_eff + q_B_eff + 3.0 * q_O)


def generate_candidates(host_df: pd.DataFrame, radii_df: pd.DataFrame, mix_ratios: List[float], mix_mode: str,
                        exclude_host_element: bool, enforce_charge_neutrality: bool, max_candidates: int,
                        verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
    lookup = radius_lookup_from_df(radii_df)
    a_df, b_df, x_df = build_substitution_sets(radii_df)

    # 固定氧半径与价态
    r_O = get_radius(lookup, "O", -2, 6)
    if r_O is None:
        raise ValueError("radii table 中缺少 O,-2,CN=6")
    q_O = -2

    # 候选替代列表（元素, 氧化态, 半径）
    a_candidates = [
        (str(r["element"]), int(r["oxidation_state"]), float(r["ionic_radius"]))
        for _, r in a_df.iterrows()
        if str(r["element"]) != "O"
    ]
    b_candidates = [
        (str(r["element"]), int(r["oxidation_state"]), float(r["ionic_radius"]))
        for _, r in b_df.iterrows()
        if str(r["element"]) != "O"
    ]

    records = []
    total_attempts = 0

    required_host_cols = [
        "material_id",
        "formula_pretty",
        "assigned_A_element",
        "assigned_B_element",
        "A_oxidation_state",
        "B_oxidation_state",
    ]
    missing = [c for c in required_host_cols if c not in host_df.columns]
    if missing:
        raise ValueError(f"宿主 CSV 缺少必要列: {missing}")

    for _, row in host_df.iterrows():
        host_id = row.get("material_id")
        host_formula = row.get("formula_pretty")
        A_host = str(row["assigned_A_element"])
        B_host = str(row["assigned_B_element"])
        q_A_host = int(row["A_oxidation_state"])
        q_B_host = int(row["B_oxidation_state"])

        r_A_host = safe_get_host_radius(row, "r_A", A_host, q_A_host, 12, lookup)
        r_B_host = safe_get_host_radius(row, "r_B", B_host, q_B_host, 6, lookup)
        r_O_host = float(row["r_O"]) if pd.notna(row.get("r_O", None)) else float(r_O)

        t_host = classical_t(r_A_host, r_B_host, r_O_host)

        # A-site mixing
        if mix_mode in ("A", "both"):
            for A_sub, q_A_sub, r_A_sub in a_candidates:
                if exclude_host_element and A_sub == A_host:
                    continue
                for x in mix_ratios:
                    total_attempts += 1
                    r_A_eff = (1.0 - x) * r_A_host + x * r_A_sub
                    r_B_eff = r_B_host
                    q_A_eff = (1.0 - x) * q_A_host + x * q_A_sub
                    q_B_eff = q_B_host

                    wtf = weighted_tf(r_A_eff, r_B_eff, r_O_host)
                    sig_A = sigma_binary(x, r_A_host, r_A_sub)
                    sig_B = 0.0
                    dq = delta_q_binary(q_A_eff, q_B_eff, q_O)

                    if enforce_charge_neutrality and dq > 1e-12:
                        continue

                    records.append({
                        "parent_material_id": host_id,
                        "parent_formula": host_formula,
                        "host_id": host_id,
                        "mix_site": "A",
                        "mix_ratio": x,
                        "candidate_formula": format_formula_A_mix(A_host, A_sub, x, B_host),
                        "A_host": A_host,
                        "A_sub": A_sub,
                        "B_host": B_host,
                        "B_sub": None,
                        "A_host_ox": q_A_host,
                        "A_sub_ox": q_A_sub,
                        "B_host_ox": q_B_host,
                        "B_sub_ox": None,
                        "r_A_host": r_A_host,
                        "r_A_sub": r_A_sub,
                        "r_B_host": r_B_host,
                        "r_B_sub": None,
                        "r_O": r_O_host,
                        "classical_t": t_host,
                        "wtf": wtf,
                        "sigma_A": sig_A,
                        "sigma_B": sig_B,
                        "delta_q": dq,
                        "r_A_eff": r_A_eff,
                        "r_B_eff": r_B_eff,
                        "q_A_eff": q_A_eff,
                        "q_B_eff": q_B_eff,
                    })
                    if 0 < max_candidates <= len(records):
                        break
                if 0 < max_candidates <= len(records):
                    break
            if 0 < max_candidates <= len(records):
                break

        # B-site mixing
        if mix_mode in ("B", "both") and not (0 < max_candidates <= len(records)):
            for B_sub, q_B_sub, r_B_sub in b_candidates:
                if exclude_host_element and B_sub == B_host:
                    continue
                for y in mix_ratios:
                    total_attempts += 1
                    r_A_eff = r_A_host
                    r_B_eff = (1.0 - y) * r_B_host + y * r_B_sub
                    q_A_eff = q_A_host
                    q_B_eff = (1.0 - y) * q_B_host + y * q_B_sub

                    wtf = weighted_tf(r_A_eff, r_B_eff, r_O_host)
                    sig_A = 0.0
                    sig_B = sigma_binary(y, r_B_host, r_B_sub)
                    dq = delta_q_binary(q_A_eff, q_B_eff, q_O)

                    if enforce_charge_neutrality and dq > 1e-12:
                        continue

                    records.append({
                        "parent_material_id": host_id,
                        "parent_formula": host_formula,
                        "host_id": host_id,
                        "mix_site": "B",
                        "mix_ratio": y,
                        "candidate_formula": format_formula_B_mix(A_host, B_host, B_sub, y),
                        "A_host": A_host,
                        "A_sub": None,
                        "B_host": B_host,
                        "B_sub": B_sub,
                        "A_host_ox": q_A_host,
                        "A_sub_ox": None,
                        "B_host_ox": q_B_host,
                        "B_sub_ox": q_B_sub,
                        "r_A_host": r_A_host,
                        "r_A_sub": None,
                        "r_B_host": r_B_host,
                        "r_B_sub": r_B_sub,
                        "r_O": r_O_host,
                        "classical_t": t_host,
                        "wtf": wtf,
                        "sigma_A": sig_A,
                        "sigma_B": sig_B,
                        "delta_q": dq,
                        "r_A_eff": r_A_eff,
                        "r_B_eff": r_B_eff,
                        "q_A_eff": q_A_eff,
                        "q_B_eff": q_B_eff,
                    })
                    if 0 < max_candidates <= len(records):
                        break
                if 0 < max_candidates <= len(records):
                    break

        if 0 < max_candidates <= len(records):
            break

    out_df = pd.DataFrame(records)

    stats = {
        "n_hosts": int(len(host_df)),
        "n_total_attempts": int(total_attempts),
        "n_generated_candidates": int(len(out_df)),
        "mix_mode": mix_mode,
        "mix_ratios": mix_ratios,
        "enforce_charge_neutrality": bool(enforce_charge_neutrality),
    }

    if verbose and len(out_df) > 0:
        print(out_df.head())

    return out_df, stats


def main():
    args = parse_args()

    if not os.path.exists(args.host_csv):
        raise FileNotFoundError(f"找不到宿主 CSV: {args.host_csv}")
    if not os.path.exists(args.radii_csv):
        raise FileNotFoundError(f"找不到 radii CSV: {args.radii_csv}")

    mix_ratios = parse_mix_ratios(args.mix_ratios)
    host_df = pd.read_csv(args.host_csv)
    radii_df = pd.read_csv(args.radii_csv)

    out_df, stats = generate_candidates(
        host_df=host_df,
        radii_df=radii_df,
        mix_ratios=mix_ratios,
        mix_mode=args.mix_mode,
        exclude_host_element=args.exclude_host_element,
        enforce_charge_neutrality=args.enforce_charge_neutrality,
        max_candidates=args.max_candidates,
        verbose=args.verbose,
    )

    out_df.to_csv(args.output_csv, index=False)
    summary_path = os.path.splitext(args.output_csv)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"已生成候选文件: {args.output_csv}")
    print(f"统计信息: {summary_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
