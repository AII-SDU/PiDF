#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 Materials Project 下载适用于 4.2 节 proof-of-concept 的 ABO3 三元氧化物数据。

功能：
1. 查询 Materials Project 中 chemsys="O-*-*" 且 formula="ABC3" 的 summary 数据
2. 保留稳定性相关字段（如 energy_above_hull, is_stable）
3. 生成二分类标签 label
4. 导出原始 CSV 和清洗后的 CSV

参考：
- 官方 getting started: 使用 mp-api 和 MPRester
- 官方 querying data: 使用 materials.summary.search(...)
- 官方 examples: "Material IDs for all ternary oxides with the form ABC3"
"""

import os
import argparse
import json
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm
from mp_api.client import MPRester


def safe_get(doc: Any, field: str, default=None):
    """兼容 emmet doc / pydantic object / dict"""
    if hasattr(doc, field):
        return getattr(doc, field)
    if isinstance(doc, dict):
        return doc.get(field, default)
    return default


def normalize_doc(doc: Any) -> Dict[str, Any]:
    """把 MP summary 文档转成普通字典"""
    row = {
        "material_id": str(safe_get(doc, "material_id")),
        "formula_pretty": safe_get(doc, "formula_pretty"),
        "energy_above_hull": safe_get(doc, "energy_above_hull"),
        "is_stable": safe_get(doc, "is_stable"),
        "formation_energy_per_atom": safe_get(doc, "formation_energy_per_atom"),
        "band_gap": safe_get(doc, "band_gap"),
        "density": safe_get(doc, "density"),
        "volume": safe_get(doc, "volume"),
        "nsites": safe_get(doc, "nsites"),
        "nelements": safe_get(doc, "nelements"),
        "elements": safe_get(doc, "elements"),
        "symmetry": safe_get(doc, "symmetry"),
        "composition_reduced": safe_get(doc, "composition_reduced"),
        "composition": safe_get(doc, "composition"),
    }

    # symmetry 可能是对象，也可能是 dict
    sym = row["symmetry"]
    if sym is not None:
        if hasattr(sym, "crystal_system"):
            row["crystal_system"] = str(sym.crystal_system)
        elif isinstance(sym, dict):
            row["crystal_system"] = sym.get("crystal_system")
        else:
            row["crystal_system"] = None

        if hasattr(sym, "symbol"):
            row["spacegroup_symbol"] = sym.symbol
        elif isinstance(sym, dict):
            row["spacegroup_symbol"] = sym.get("symbol")
        else:
            row["spacegroup_symbol"] = None

        if hasattr(sym, "number"):
            row["spacegroup_number"] = sym.number
        elif isinstance(sym, dict):
            row["spacegroup_number"] = sym.get("number")
        else:
            row["spacegroup_number"] = None
    else:
        row["crystal_system"] = None
        row["spacegroup_symbol"] = None
        row["spacegroup_number"] = None

    # composition_reduced 可能是对象或 dict
    comp_red = row["composition_reduced"]
    if comp_red is not None:
        try:
            # pymatgen Composition 对象
            row["reduced_formula"] = comp_red.reduced_formula
            row["reduced_dict"] = comp_red.as_dict()
        except Exception:
            if isinstance(comp_red, dict):
                row["reduced_formula"] = None
                row["reduced_dict"] = comp_red
            else:
                row["reduced_formula"] = None
                row["reduced_dict"] = None
    else:
        row["reduced_formula"] = None
        row["reduced_dict"] = None

    # elements 统一转成字符串列表
    if row["elements"] is not None:
        try:
            row["elements"] = [str(x) for x in row["elements"]]
        except Exception:
            pass

    return row


def is_strict_abo3_oxide(row: pd.Series) -> bool:
    """
    严格筛 ABO3 三元氧化物：
    - 仅 3 种元素
    - 包含 O
    - reduced composition 中 O 的计量数为 3
    - 另外两个元素的计量数各为 1
    """
    if row.get("nelements", None) != 3:
        return False

    red = row.get("reduced_dict", None)
    if not isinstance(red, dict):
        return False

    # 统一成 float
    red = {str(k): float(v) for k, v in red.items()}

    if "O" not in red:
        return False

    if len(red) != 3:
        return False

    if abs(red["O"] - 3.0) > 1e-8:
        return False

    others = [v for k, v in red.items() if k != "O"]
    if len(others) != 2:
        return False

    return all(abs(v - 1.0) < 1e-8 for v in others)


def assign_label(
    row: pd.Series,
    label_mode: str = "is_stable",
    ehull_threshold: float = 0.0,
) -> int:
    """
    生成二分类标签：
    - is_stable: 直接使用 MP 的 is_stable
    - ehull: 使用 energy_above_hull <= threshold
    """
    if label_mode == "is_stable":
        val = row.get("is_stable", None)
        if pd.isna(val):
            return -1
        return int(bool(val))

    if label_mode == "ehull":
        e = row.get("energy_above_hull", None)
        if pd.isna(e):
            return -1
        return int(float(e) <= ehull_threshold)

    raise ValueError("label_mode 只能是 'is_stable' 或 'ehull'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default="AM8UHtnYfxNHBLaWJjorsvIhcshVGa5T")
    parser.add_argument("--outdir", type=str, default="mp_data")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="is_stable",
        choices=["is_stable", "ehull"],
        help="标签生成方式：直接用 is_stable，或用 energy_above_hull 阈值",
    )
    parser.add_argument(
        "--ehull-threshold",
        type=float,
        default=0.0,
        help="当 label-mode=ehull 时使用的阈值（单位通常为 eV/atom）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="保留参数，当前 summary.search 直接整批拉取；如后续扩展可用于分块",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError(
            "没有检测到 Materials Project API key。"
            "请使用 --api-key 传入，或先设置环境变量 MP_API_KEY。"
        )

    os.makedirs(args.outdir, exist_ok=True)

    fields = [
        "material_id",
        "formula_pretty",
        "energy_above_hull",
        "is_stable",
        "formation_energy_per_atom",
        "band_gap",
        "density",
        "volume",
        "nsites",
        "nelements",
        "elements",
        "symmetry",
        "composition_reduced",
        "composition",
    ]

    print("开始从 Materials Project 下载 ABC3 三元氧化物 summary 数据...")

    with MPRester(args.api_key) as mpr:
        docs = mpr.materials.summary.search(
            chemsys="O-*-*",
            formula="ABC3",
            fields=fields,
        )

    print(f"下载完成，原始文档数：{len(docs)}")

    rows: List[Dict[str, Any]] = []
    for doc in tqdm(docs, desc="规范化文档"):
        rows.append(normalize_doc(doc))

    raw_df = pd.DataFrame(rows)
    raw_csv = os.path.join(args.outdir, "mp_abo3_raw.csv")
    raw_df.to_csv(raw_csv, index=False)
    print(f"原始 CSV 已保存到：{raw_csv}")

    print("开始进行严格 ABO3 氧化物过滤...")
    clean_df = raw_df.copy()
    clean_df = clean_df[clean_df.apply(is_strict_abo3_oxide, axis=1)].reset_index(drop=True)

    print(f"严格 ABO3 过滤后样本数：{len(clean_df)}")

    print("生成二分类标签...")
    clean_df["label"] = clean_df.apply(
        lambda row: assign_label(
            row,
            label_mode=args.label_mode,
            ehull_threshold=args.ehull_threshold,
        ),
        axis=1,
    )

    clean_df = clean_df[clean_df["label"] >= 0].reset_index(drop=True)

    # 生成一个简单 host_id，4.2 做 group split 时可先用 reduced_formula 占位
    # 后续如果你自己定义 parent host，可替换这里
    clean_df["host_id"] = clean_df["reduced_formula"].fillna(clean_df["formula_pretty"])

    # 把 reduced_dict 展平成 A, B, O 三列，便于后续人工检查
    A_list, B_list, O_list = [], [], []
    for _, row in clean_df.iterrows():
        red = row["reduced_dict"]
        if isinstance(red, str):
            # 如果 CSV 读写后变成字符串，这里就不再解析
            A_list.append(None)
            B_list.append(None)
            O_list.append(None)
            continue

        if isinstance(red, dict):
            elems = [k for k in red.keys() if k != "O"]
            if len(elems) == 2:
                A_list.append(elems[0])
                B_list.append(elems[1])
                O_list.append(red.get("O"))
            else:
                A_list.append(None)
                B_list.append(None)
                O_list.append(None)
        else:
            A_list.append(None)
            B_list.append(None)
            O_list.append(None)

    clean_df["A_element"] = A_list
    clean_df["B_element"] = B_list
    clean_df["O_count"] = O_list

    clean_csv = os.path.join(args.outdir, "mp_abo3_clean.csv")
    clean_df.to_csv(clean_csv, index=False)
    print(f"清洗后 CSV 已保存到：{clean_csv}")

    stats = {
        "raw_count": int(len(raw_df)),
        "strict_abo3_count": int(len(clean_df)),
        "stable_count": int((clean_df["label"] == 1).sum()),
        "unstable_count": int((clean_df["label"] == 0).sum()),
        "label_mode": args.label_mode,
        "ehull_threshold": args.ehull_threshold,
    }

    stats_path = os.path.join(args.outdir, "download_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("数据下载与清洗完成。")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()