"""Co-primary P3 order-information comparisons for Panel B."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.main_analysis_report import bootstrap_ci, fmt


def order_information_bounds(bounds: pd.DataFrame) -> pd.DataFrame:
    """Conservative bounds for the exacta-minus-quinella Harville improvement.

    For market m, Harville improvement is I_m = TV_H,m - TV_M,m. With only
    TV_H,m in [L_H,m,U_H,m] and TV_M,m in [L_M,m,U_M,m], a valid lower bound
    for I_exacta-I_quinella is

      L_H,E - U_M,E - U_H,Q + L_M,Q.

    The corresponding upper bound reverses every endpoint. The lower-bound
    median is the pre-registered conservative Panel B decision statistic.
    """
    lower = bounds.pivot(
        index=["race_id", "target_market"], columns="model", values="tv_lower"
    )
    upper = bounds.pivot(
        index=["race_id", "target_market"], columns="model", values="tv_upper_outer"
    )
    e_l = lower.xs("exacta", level="target_market")
    e_u = upper.xs("exacta", level="target_market")
    q_l = lower.xs("quinella", level="target_market")
    q_u = upper.xs("quinella", level="target_market")
    common = e_l.index.intersection(e_u.index).intersection(q_l.index).intersection(q_u.index)
    required = {"main", "harville"}
    if not required.issubset(e_l.columns) or not required.issubset(q_l.columns):
        raise ValueError("Panel B P3 requires main and Harville bounds for exacta and quinella")

    lower_diff = (
        e_l.loc[common, "harville"]
        - e_u.loc[common, "main"]
        - q_u.loc[common, "harville"]
        + q_l.loc[common, "main"]
    ).to_numpy(dtype=float)
    upper_diff = (
        e_u.loc[common, "harville"]
        - e_l.loc[common, "main"]
        - q_l.loc[common, "harville"]
        + q_u.loc[common, "main"]
    ).to_numpy(dtype=float)
    if np.any(lower_diff - upper_diff > 1e-10):
        raise RuntimeError("Panel B P3 bounds are inverted")

    lower_med, lower_ci_low, lower_ci_high = bootstrap_ci(
        lower_diff, label="B|P3|difference_lower"
    )
    upper_med, upper_ci_low, upper_ci_high = bootstrap_ci(
        upper_diff, label="B|P3|difference_upper"
    )
    return pd.DataFrame(
        [
            {
                "panel": "B",
                "measure": "tv",
                "n_races": len(common),
                "median_difference_lower": lower_med,
                "difference_lower_ci_low": lower_ci_low,
                "difference_lower_ci_high": lower_ci_high,
                "median_difference_upper": upper_med,
                "difference_upper_ci_low": upper_ci_low,
                "difference_upper_ci_high": upper_ci_high,
                "robust_positive_difference": bool(
                    np.isfinite(lower_ci_low) and lower_ci_low > 0
                ),
            }
        ]
    )


def write_order_information_bounds_table(table_dir: Path, summary: pd.DataFrame) -> Path:
    table_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"거리 & 경주 수 & 차이 하한 & 하한 95\% CI & 차이 상한 & 상한 95\% CI & 보수적 양(+) 판정 \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{fmt(row['measure'])} & {fmt(row['n_races'])} & "
            f"{fmt(row['median_difference_lower'])} & "
            f"[{fmt(row['difference_lower_ci_low'])}, {fmt(row['difference_lower_ci_high'])}] & "
            f"{fmt(row['median_difference_upper'])} & "
            f"[{fmt(row['difference_upper_ci_low'])}, {fmt(row['difference_upper_ci_high'])}] & "
            f"{fmt(row['robust_positive_difference'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = table_dir / "main_order_information_bounds.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
