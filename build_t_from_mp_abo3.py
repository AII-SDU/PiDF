#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 mp_abo3_clean.csv 继续生成 classical_t 和 wtf。

注意：
1. 该脚本针对单组分 ABO3 宿主数据。
2. 在这种输入下，wtf 会退化为 classical_t。
3. 需要用户提供 ionic_radii.csv（Shannon 半径表的简化版本）。
4. A/B 位是通过启发式规则推断的；如果你已经手工确定了 A/B，
   可以直接在输入 CSV 中提供 A_element 和 B_element 列，并使用 --trust-ab-columns。
"""

import os
import ast
import math
import json
import argparse
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=str,
        default="./mp_data/mp_abo3_clean.csv",
        help="输入的 mp_abo3_clean.csv 路径",
    )
    parser.add_argument(
        "--radii-csv",
        type=str,
        default="./mp_data/ionic_radii.csv",
        help="离子半径表 CSV 路径，需包含 element, oxidation_state, coordination, ionic_radius",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./mp_data/mp_abo3_with_t.csv",
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--trust-ab-columns",
        action="store_true",
        help="若输入 CSV 已经有可信的 A_element / B_element，则直接使用",
    )
    parser.add_argument(
        "--default-a-cn",
        type=int,
        default=12,
        help="A 位默认配位数",
    )
    parser.add_argument(
        "--default-b-cn",
        type=int,
        default=6,
        help="B 位默认配位数",
    )
    parser.add_argument(
        "--oxygen-cn",
        type=int,
        default=6,
        help="氧离子半径使用的配位数",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细日志",
    )
    return parser.parse_args()


def load_radii_table(radii_csv: str) -> pd.DataFrame:
    df = pd.read_csv(radii_csv)
    required = {"element", "oxidation_state", "coordination", "ionic_radius"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"radii_csv 缺少必要列: {missing}")
    df["element"] = df["element"].astype(str)
    df["oxidation_state"] = df["oxidation_state"].astype(int)
    df["coordination"] = df["coordination"].astype(int)
    df["ionic_radius"] = df["ionic_radius"].astype(float)
    return df


def get_radius_lookup(radii_df: pd.DataFrame) -> Dict[Tuple[str, int, int], float]:
    lookup = {}
    for _, row in radii_df.iterrows():
        key = (row["element"], int(row["oxidation_state"]), int(row["coordination"]))
        lookup[key] = float(row["ionic_radius"])
    return lookup


def available_states_for_element(
    radii_df: pd.DataFrame,
    element: str,
    coordination: int,
) -> List[int]:
    sub = radii_df[
        (radii_df["element"] == element) &
        (radii_df["coordination"] == coordination)
    ]
    states = sorted(sub["oxidation_state"].dropna().astype(int).unique().tolist())
    return states


def get_radius(
    lookup: Dict[Tuple[str, int, int], float],
    element: str,
    oxidation_state: int,
    coordination: int,
) -> Optional[float]:
    return lookup.get((element, int(oxidation_state), int(coordination)), None)


def parse_reduced_dict(value) -> Dict[str, float]:
    """
    兼容 dict 或 CSV 读回来后的字符串形式。
    """
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if isinstance(value, str):
        try:
            obj = ast.literal_eval(value)
            if isinstance(obj, dict):
                return {str(k): float(v) for k, v in obj.items()}
        except Exception:
            pass
    return {}


def extract_two_cations(row: pd.Series, trust_ab_columns: bool) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    返回 (A_candidate_or_A, B_candidate_or_B, cation_list)
    - 若 trust_ab_columns=True 且存在 A_element/B_element，则优先使用
    - 否则从 reduced_dict 中提取两个非 O 元素
    """
    if trust_ab_columns and ("A_element" in row.index) and ("B_element" in row.index):
        a = row.get("A_element", None)
        b = row.get("B_element", None)
        if pd.notna(a) and pd.notna(b):
            return str(a), str(b), [str(a), str(b)]

    red = parse_reduced_dict(row.get("reduced_dict", None))
    cations = [el for el in red.keys() if el != "O"]

    if len(cations) != 2:
        return None, None, cations

    return None, None, cations


def score_assignment(r_A: float, r_B: float, t: float) -> float:
    """
    启发式打分：
    1. A 位半径应大于 B 位半径
    2. t 更接近常见钙钛矿区间更优
    """
    score = 0.0

    # A 应明显比 B 大
    if r_A > r_B:
        score += 3.0
    else:
        score -= 5.0

    # 更偏好落在经验合理区间内
    if 0.75 <= t <= 1.10:
        score += 2.0

    # 偏好更接近 1
    score -= abs(t - 1.0)

    return score


def infer_ab_assignment_and_t(
    cation1: str,
    cation2: str,
    radii_df: pd.DataFrame,
    radius_lookup: Dict[Tuple[str, int, int], float],
    a_cn: int,
    b_cn: int,
    o_cn: int,
) -> Dict:
    """
    推断：
    - 哪个元素是 A，哪个是 B
    - 对应氧化态
    - 经典 tolerance factor

    电中性条件：q_A + q_B = 6 （因为 ABO3, O 为 -2）
    """
    o_radius = get_radius(radius_lookup, "O", -2, o_cn)
    if o_radius is None:
        raise ValueError(f"半径表中缺少 O, -2, CN={o_cn} 的数据")

    candidates = []

    # 两种元素分配方式：cation1->A, cation2->B 或反过来
    assignments = [
        (cation1, cation2),
        (cation2, cation1),
    ]

    for a_el, b_el in assignments:
        a_states = available_states_for_element(radii_df, a_el, a_cn)
        b_states = available_states_for_element(radii_df, b_el, b_cn)

        for qA in a_states:
            for qB in b_states:
                # ABO3 电中性
                if qA + qB != 6:
                    continue
                if qA <= 0 or qB <= 0:
                    continue

                r_A = get_radius(radius_lookup, a_el, qA, a_cn)
                r_B = get_radius(radius_lookup, b_el, qB, b_cn)

                if r_A is None or r_B is None:
                    continue

                t = (r_A + o_radius) / (math.sqrt(2.0) * (r_B + o_radius))
                s = score_assignment(r_A, r_B, t)

                candidates.append({
                    "A_element": a_el,
                    "B_element": b_el,
                    "A_oxi": qA,
                    "B_oxi": qB,
                    "r_A": r_A,
                    "r_B": r_B,
                    "r_O": o_radius,
                    "classical_t": t,
                    "score": s,
                })

    if not candidates:
        return {
            "success": False,
            "reason": f"无法为 {cation1}-{cation2} 找到满足 ABO3 电中性的 A/B 分配与配位半径"
        }

    best = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]
    best["success"] = True
    best["reason"] = "ok"
    return best


def process_dataframe(
    df: pd.DataFrame,
    radii_df: pd.DataFrame,
    trust_ab_columns: bool,
    a_cn: int,
    b_cn: int,
    o_cn: int,
    verbose: bool = False,
) -> pd.DataFrame:
    radius_lookup = get_radius_lookup(radii_df)

    out_rows = []
    fail_count = 0

    for idx, row in df.iterrows():
        preset_a, preset_b, cation_list = extract_two_cations(row, trust_ab_columns=trust_ab_columns)

        if preset_a is not None and preset_b is not None:
            c1, c2 = preset_a, preset_b
        else:
            if len(cation_list) != 2:
                fail_count += 1
                new_row = row.to_dict()
                new_row.update({
                    "assigned_A_element": None,
                    "assigned_B_element": None,
                    "A_oxidation_state": None,
                    "B_oxidation_state": None,
                    "r_A": None,
                    "r_B": None,
                    "r_O": None,
                    "classical_t": None,
                    "wtf": None,
                    "descriptor_note": "无法从 reduced_dict 提取两个非 O 阳离子",
                })
                out_rows.append(new_row)
                continue
            c1, c2 = cation_list[0], cation_list[1]

        result = infer_ab_assignment_and_t(
            cation1=c1,
            cation2=c2,
            radii_df=radii_df,
            radius_lookup=radius_lookup,
            a_cn=a_cn,
            b_cn=b_cn,
            o_cn=o_cn,
        )

        new_row = row.to_dict()

        if not result["success"]:
            fail_count += 1
            new_row.update({
                "assigned_A_element": None,
                "assigned_B_element": None,
                "A_oxidation_state": None,
                "B_oxidation_state": None,
                "r_A": None,
                "r_B": None,
                "r_O": None,
                "classical_t": None,
                "wtf": None,
                "descriptor_note": result["reason"],
            })
            out_rows.append(new_row)
            if verbose:
                print(f"[WARN] row={idx}: {result['reason']}")
            continue

        classical_t = result["classical_t"]

        # 关键说明：
        # 当前 mp_abo3_clean.csv 是单组分 ABO3 宿主数据，
        # 因此 weighted tolerance factor 退化为 classical_t
        wtf = classical_t

        new_row.update({
            "assigned_A_element": result["A_element"],
            "assigned_B_element": result["B_element"],
            "A_oxidation_state": result["A_oxi"],
            "B_oxidation_state": result["B_oxi"],
            "r_A": result["r_A"],
            "r_B": result["r_B"],
            "r_O": result["r_O"],
            "classical_t": classical_t,
            "wtf": wtf,
            "descriptor_note": "单组分 ABO3 宿主，wtf 退化为 classical_t",
        })
        out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows)

    summary = {
        "total_rows": int(len(df)),
        "success_rows": int(out_df["classical_t"].notna().sum()),
        "failed_rows": int(fail_count),
    }
    return out_df, summary


def main():
    args = parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"找不到输入文件: {args.input_csv}")
    if not os.path.exists(args.radii_csv):
        raise FileNotFoundError(f"找不到半径表文件: {args.radii_csv}")

    df = pd.read_csv(args.input_csv)
    radii_df = load_radii_table(args.radii_csv)

    out_df, summary = process_dataframe(
        df=df,
        radii_df=radii_df,
        trust_ab_columns=args.trust_ab_columns,
        a_cn=args.default_a_cn,
        b_cn=args.default_b_cn,
        o_cn=args.oxygen_cn,
        verbose=args.verbose,
    )

    # 按是否成功算出 classical_t 分开保存
    success_df = out_df[out_df["classical_t"].notna()].copy()
    failed_df = out_df[out_df["classical_t"].isna()].copy()

    # 成功样本保存到主输出文件
    success_df.to_csv(args.output_csv, index=False)

    # 失败样本单独保存，便于排查
    failed_csv = os.path.splitext(args.output_csv)[0] + "_failed_hosts.csv"
    failed_df.to_csv(failed_csv, index=False)

    # 更新 summary
    summary["success_rows"] = int(len(success_df))
    summary["failed_rows"] = int(len(failed_df))
    summary["success_output_csv"] = args.output_csv
    summary["failed_output_csv"] = failed_csv

    summary_path = os.path.splitext(args.output_csv)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("处理完成")
    print(f"成功样本输出文件: {args.output_csv}")
    print(f"失败样本输出文件: {failed_csv}")
    print(f"统计文件: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()