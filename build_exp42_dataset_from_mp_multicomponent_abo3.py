#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import argparse
from itertools import combinations
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
from tqdm import tqdm
from pymatgen.core import Composition
from mp_api.client import MPRester


def safe_get(doc: Any, field: str, default=None):
    if hasattr(doc, field):
        return getattr(doc, field)
    if isinstance(doc, dict):
        return doc.get(field, default)
    return default


def load_radii_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"element", "oxidation_state", "coordination", "ionic_radius", "site_group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ionic_radii.csv 缺少必要列: {missing}")
    df["element"] = df["element"].astype(str)
    df["oxidation_state"] = df["oxidation_state"].astype(int)
    df["coordination"] = df["coordination"].astype(int)
    df["ionic_radius"] = df["ionic_radius"].astype(float)
    df["site_group"] = df["site_group"].astype(str)
    return df


def build_radii_lookup(radii_df: pd.DataFrame) -> Dict[Tuple[str, int, int], float]:
    lookup = {}
    for _, row in radii_df.iterrows():
        lookup[(row["element"], int(row["oxidation_state"]), int(row["coordination"]))] = float(row["ionic_radius"])
    return lookup


def build_state_lookup(radii_df: pd.DataFrame) -> Dict[str, int]:
    """
    这里假设每个元素在工作表里只保留一个代表氧化态。
    """
    state_lookup = {}
    for _, row in radii_df.iterrows():
        el = row["element"]
        ox = int(row["oxidation_state"])
        if el in state_lookup and state_lookup[el] != ox:
            raise ValueError(f"元素 {el} 在 ionic_radii.csv 中存在多个氧化态，当前脚本假设每个元素只有一个代表氧化态。")
        state_lookup[el] = ox
    return state_lookup


def get_site_sets(radii_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    A_set = sorted(radii_df[radii_df["site_group"] == "A"]["element"].unique().tolist())
    B_set = sorted(radii_df[radii_df["site_group"] == "B"]["element"].unique().tolist())
    return A_set, B_set


def get_radius(lookup: Dict[Tuple[str, int, int], float], el: str, ox: int, cn: int) -> Optional[float]:
    return lookup.get((el, int(ox), int(cn)), None)


def generate_chemsys_queries(A_set: List[str], B_set: List[str]) -> List[Tuple[str, str]]:
    """
    返回列表 [(mix_site, chemsys), ...]
    mix_site in {"A", "B"}
    """
    queries = []

    # A-site mixed: A1-A2-B-O
    for a1, a2 in combinations(A_set, 2):
        for b in B_set:
            chemsys = "-".join(sorted([a1, a2, b, "O"]))
            queries.append(("A", chemsys))

    # B-site mixed: A-B1-B2-O
    for a in A_set:
        for b1, b2 in combinations(B_set, 2):
            chemsys = "-".join(sorted([a, b1, b2, "O"]))
            queries.append(("B", chemsys))

    # 去重
    queries = list(dict.fromkeys(queries))
    return queries


def normalize_to_abo3(comp: Composition) -> Dict[str, float]:
    """
    把任意组成按 O=3 归一化。
    例如 KNaTa2O6 -> K0.5 Na0.5 Ta1 O3
    """
    comp_dict = comp.as_dict()
    if "O" not in comp_dict:
        raise ValueError("组成中不含 O")
    o_amt = float(comp_dict["O"])
    if o_amt <= 0:
        raise ValueError("O 的计量数非法")
    scale = 3.0 / o_amt
    norm = {str(el): float(v) * scale for el, v in comp_dict.items()}
    return norm


def classify_multicomponent_abo3(
    norm_comp: Dict[str, float],
    A_set: List[str],
    B_set: List[str],
    tol: float = 1e-6,
) -> Optional[Dict[str, Any]]:
    """
    输入已经归一化到 O=3 的组成字典。
    判断是否属于：
    - A-site mixed: A1_x A2_(1-x) B O3
    - B-site mixed: A B1_y B2_(1-y) O3
    """
    if "O" not in norm_comp:
        return None
    if abs(norm_comp["O"] - 3.0) > tol:
        return None

    cations = {k: v for k, v in norm_comp.items() if k != "O"}
    if len(cations) != 3:
        return None

    A_like = [(el, amt) for el, amt in cations.items() if el in A_set]
    B_like = [(el, amt) for el, amt in cations.items() if el in B_set]

    # 总 A 位计量和 B 位计量都应接近 1
    A_total = sum(v for _, v in A_like)
    B_total = sum(v for _, v in B_like)

    if abs(A_total - 1.0) > 1e-5 or abs(B_total - 1.0) > 1e-5:
        return None

    # A-site mixed: 两个 A-like + 一个 B-like
    if len(A_like) == 2 and len(B_like) == 1:
        A_sorted = sorted(A_like, key=lambda x: x[0])
        B_sorted = sorted(B_like, key=lambda x: x[0])
        return {
            "mix_site": "A",
            "A_elements": [A_sorted[0][0], A_sorted[1][0]],
            "A_amounts": [A_sorted[0][1], A_sorted[1][1]],
            "B_elements": [B_sorted[0][0]],
            "B_amounts": [B_sorted[0][1]],
        }

    # B-site mixed: 一个 A-like + 两个 B-like
    if len(A_like) == 1 and len(B_like) == 2:
        A_sorted = sorted(A_like, key=lambda x: x[0])
        B_sorted = sorted(B_like, key=lambda x: x[0])
        return {
            "mix_site": "B",
            "A_elements": [A_sorted[0][0]],
            "A_amounts": [A_sorted[0][1]],
            "B_elements": [B_sorted[0][0], B_sorted[1][0]],
            "B_amounts": [B_sorted[0][1], B_sorted[1][1]],
        }

    return None


def compute_descriptors(
    info: Dict[str, Any],
    radius_lookup: Dict[Tuple[str, int, int], float],
    ox_lookup: Dict[str, int],
    a_cn: int = 12,
    b_cn: int = 6,
    o_cn: int = 6,
) -> Optional[Dict[str, Any]]:
    """
    计算 classical_t, wtf, sigma_A, sigma_B, delta_q
    classical_t 定义为：忽略混合时，使用混合位点上的“主导物种”作为 classical baseline
    """
    r_O = get_radius(radius_lookup, "O", -2, o_cn)
    if r_O is None:
        raise ValueError("ionic_radii.csv 中缺少 O,-2,6 的半径")

    # A site
    A_elements = info["A_elements"]
    A_amounts = info["A_amounts"]
    B_elements = info["B_elements"]
    B_amounts = info["B_amounts"]

    # 计算有效半径
    rA_list = []
    qA_list = []
    for el, amt in zip(A_elements, A_amounts):
        q = ox_lookup[el]
        r = get_radius(radius_lookup, el, q, a_cn)
        if r is None:
            return None
        rA_list.append((el, amt, r))
        qA_list.append((el, amt, q))

    rB_list = []
    qB_list = []
    for el, amt in zip(B_elements, B_amounts):
        q = ox_lookup[el]
        r = get_radius(radius_lookup, el, q, b_cn)
        if r is None:
            return None
        rB_list.append((el, amt, r))
        qB_list.append((el, amt, q))

    r_A_eff = sum(amt * r for _, amt, r in rA_list)
    r_B_eff = sum(amt * r for _, amt, r in rB_list)

    q_A_eff = sum(amt * q for _, amt, q in qA_list)
    q_B_eff = sum(amt * q for _, amt, q in qB_list)

    wtf = (r_A_eff + r_O) / (math.sqrt(2.0) * (r_B_eff + r_O))

    sigma_A = math.sqrt(sum(amt * (r - r_A_eff) ** 2 for _, amt, r in rA_list))
    sigma_B = math.sqrt(sum(amt * (r - r_B_eff) ** 2 for _, amt, r in rB_list))

    delta_q = abs(q_A_eff + q_B_eff + 3 * (-2))

    # classical baseline: 使用混合位点上的主导物种，未混合位点用唯一物种
    if info["mix_site"] == "A":
        A_dom = sorted(rA_list, key=lambda x: x[1], reverse=True)[0]
        r_A_classical = A_dom[2]
        r_B_classical = rB_list[0][2]
    else:
        r_A_classical = rA_list[0][2]
        B_dom = sorted(rB_list, key=lambda x: x[1], reverse=True)[0]
        r_B_classical = B_dom[2]

    classical_t = (r_A_classical + r_O) / (math.sqrt(2.0) * (r_B_classical + r_O))

    return {
        "classical_t": classical_t,
        "wtf": wtf,
        "sigma_A": sigma_A,
        "sigma_B": sigma_B,
        "delta_q": delta_q,
        "r_A_eff": r_A_eff,
        "r_B_eff": r_B_eff,
        "q_A_eff": q_A_eff,
        "q_B_eff": q_B_eff,
        "r_O": r_O,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radii-csv", type=str, default="./mp_data/ionic_radii.csv", help="ionic_radii.csv 路径")
    parser.add_argument("--output-csv", type=str, default="./mp_data/exp42_mp_multicomponent_abo3.csv")
    parser.add_argument("--stats-json", type=str, default="./mp_data/exp42_mp_multicomponent_abo3_stats.json")
    parser.add_argument("--api-key", type=str, default="AM8UHtnYfxNHBLaWJjorsvIhcshVGa5T")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="ehull",
        choices=["is_stable", "ehull"],
        help="直接使用 is_stable 或按 energy_above_hull 阈值生成 label",
    )
    parser.add_argument("--ehull-threshold", type=float, default=0.05)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("未检测到 MP_API_KEY，请设置环境变量或通过 --api-key 传入")

    radii_df = load_radii_table(args.radii_csv)
    radius_lookup = build_radii_lookup(radii_df)
    ox_lookup = build_state_lookup(radii_df)
    A_set, B_set = get_site_sets(radii_df)

    queries = generate_chemsys_queries(A_set, B_set)

    # 逐个 chemsys 下载 summary 数据
    fields = [
        "material_id",
        "formula_pretty",
        "energy_above_hull",
        "is_stable",
        "composition_reduced",
        "composition",
        "elements",
        "nelements",
        "symmetry",
    ]

    all_docs = []
    with MPRester(args.api_key) as mpr:
        for mix_site_hint, chemsys in tqdm(queries, desc="Downloading MP docs by chemsys"):
            try:
                docs = mpr.materials.summary.search(
                    chemsys=chemsys,
                    fields=fields,
                )
            except Exception:
                docs = []

            for doc in docs:
                all_docs.append((mix_site_hint, chemsys, doc))
            time.sleep(args.sleep)

    rows = []
    seen_mpids = set()

    for mix_site_hint, chemsys, doc in tqdm(all_docs, desc="Filtering and building 4.2 dataset"):
        mpid = str(safe_get(doc, "material_id"))
        if mpid in seen_mpids:
            continue
        seen_mpids.add(mpid)

        comp_red = safe_get(doc, "composition_reduced", None)
        comp_full = safe_get(doc, "composition", None)

        try:
            if comp_red is not None:
                comp = comp_red if isinstance(comp_red, Composition) else Composition(comp_red)
            elif comp_full is not None:
                comp = comp_full if isinstance(comp_full, Composition) else Composition(comp_full)
            else:
                continue
        except Exception:
            continue

        # 只保留四元体系（两种 A 或两种 B + 另一个位点 + O）
        elements = sorted([str(el) for el in comp.elements])
        if len(elements) != 4 or "O" not in elements:
            continue

        try:
            norm_comp = normalize_to_abo3(comp)
        except Exception:
            continue

        classified = classify_multicomponent_abo3(norm_comp, A_set, B_set)
        if classified is None:
            continue

        desc = compute_descriptors(classified, radius_lookup, ox_lookup)
        if desc is None:
            continue

        energy_above_hull = safe_get(doc, "energy_above_hull", None)
        is_stable = safe_get(doc, "is_stable", None)

        if args.label_mode == "is_stable":
            label = None if is_stable is None else int(bool(is_stable))
        else:
            label = None if energy_above_hull is None else int(float(energy_above_hull) <= args.ehull_threshold)

        # 构造便于后续 4.2 使用的统一列
        row = {
            "material_id": mpid,
            "formula_pretty": safe_get(doc, "formula_pretty", None),
            "chemsys_query": chemsys,
            "host_id": chemsys,  # 4.2 若做 group split，可先按 chemical system 分组
            "mix_site": classified["mix_site"],
            "label": label,
            "is_stable": is_stable,
            "energy_above_hull": energy_above_hull,
            "classical_t": desc["classical_t"],
            "wtf": desc["wtf"],
            "sigma_A": desc["sigma_A"],
            "sigma_B": desc["sigma_B"],
            "delta_q": desc["delta_q"],
            "r_A_eff": desc["r_A_eff"],
            "r_B_eff": desc["r_B_eff"],
            "q_A_eff": desc["q_A_eff"],
            "q_B_eff": desc["q_B_eff"],
            "A_elements": json.dumps(classified["A_elements"], ensure_ascii=False),
            "A_amounts": json.dumps(classified["A_amounts"], ensure_ascii=False),
            "B_elements": json.dumps(classified["B_elements"], ensure_ascii=False),
            "B_amounts": json.dumps(classified["B_amounts"], ensure_ascii=False),
            "normalized_composition": json.dumps(norm_comp, ensure_ascii=False),
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)

    # 去掉没有 label 的行
    out_df = out_df[out_df["label"].notna()].copy()
    out_df["label"] = out_df["label"].astype(int)

    out_df.to_csv(args.output_csv, index=False)

    stats = {
        "total_rows": int(len(out_df)),
        "stable_rows": int((out_df["label"] == 1).sum()),
        "unstable_rows": int((out_df["label"] == 0).sum()),
        "n_A_mixed": int((out_df["mix_site"] == "A").sum()) if len(out_df) > 0 else 0,
        "n_B_mixed": int((out_df["mix_site"] == "B").sum()) if len(out_df) > 0 else 0,
        "label_mode": args.label_mode,
        "ehull_threshold": args.ehull_threshold,
        "n_unique_chemsys_queries": int(len(queries)),
    }

    with open(args.stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("4.2 数据集构建完成")
    print(f"输出文件: {args.output_csv}")
    print(f"统计文件: {args.stats_json}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()