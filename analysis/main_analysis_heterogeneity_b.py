"""Panel B heterogeneity summaries and manuscript comparison table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGETS = ("exacta", "quinella", "trio", "win")
DIMENSIONS = ("year", "meet", "n_valid_horses")


def panel_b_heterogeneity(bounds: pd.DataFrame) -> pd.DataFrame:
    required = {"race_id", "target_market", "model", "n_valid_horses", "tv_lower", "tv_upper_outer"}
    missing = required.difference(bounds.columns)
    if missing:
        raise ValueError(f"Panel B bounds missing columns: {sorted(missing)}")
    frame = bounds.copy()
    frame["year"] = frame["race_id"].astype(str).str.slice(0, 4).astype(int)
    frame["meet"] = frame["race_id"].astype(str).str.split("_").str[1].astype(int)
    pieces: list[pd.DataFrame] = []
    for dimension in DIMENSIONS:
        grouped = (
            frame.groupby(["target_market", "model", dimension], dropna=False)
            .agg(
                n_races=("tv_lower", "size"),
                median_tv_lower=("tv_lower", "median"),
                q25_tv_lower=("tv_lower", lambda x: x.quantile(0.25)),
                q75_tv_lower=("tv_lower", lambda x: x.quantile(0.75)),
                median_tv_upper_outer=("tv_upper_outer", "median"),
                q25_tv_upper_outer=("tv_upper_outer", lambda x: x.quantile(0.25)),
                q75_tv_upper_outer=("tv_upper_outer", lambda x: x.quantile(0.75)),
            )
            .reset_index()
        )
        grouped["dimension"] = dimension
        grouped["level"] = grouped[dimension].astype(str)
        pieces.append(
            grouped[
                [
                    "target_market", "model", "dimension", "level", "n_races",
                    "median_tv_lower", "q25_tv_lower", "q75_tv_lower",
                    "median_tv_upper_outer", "q25_tv_upper_outer", "q75_tv_upper_outer",
                ]
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def comparison_summary(panel_a: pd.DataFrame, panel_b: pd.DataFrame) -> pd.DataFrame:
    a = panel_a[panel_a["model"].eq("main")]
    b = panel_b[panel_b["model"].eq("main")]
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for dimension in DIMENSIONS:
            aa = a[a["target_market"].eq(target) & a["dimension"].eq(dimension)]
            bb = b[b["target_market"].eq(target) & b["dimension"].eq(dimension)]
            if aa.empty or bb.empty:
                raise ValueError(f"missing heterogeneity cell: {target}/{dimension}")
            rows.append(
                {
                    "target_market": target,
                    "dimension": dimension,
                    "panel_a_median_tv_min": float(aa["median_tv"].min()),
                    "panel_a_median_tv_max": float(aa["median_tv"].max()),
                    "panel_b_median_tv_lower_min": float(bb["median_tv_lower"].min()),
                    "panel_b_median_tv_lower_max": float(bb["median_tv_lower"].max()),
                    "panel_b_median_tv_upper_min": float(bb["median_tv_upper_outer"].min()),
                    "panel_b_median_tv_upper_max": float(bb["median_tv_upper_outer"].max()),
                }
            )
    return pd.DataFrame(rows)


def render_table(summary: pd.DataFrame, output: Path) -> None:
    target_label = {"exacta": "쌍승", "quinella": "복승", "trio": "삼복승"}
    dimension_label = {"year": "연도", "meet": "경마장", "n_valid_horses": "출전두수"}
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"승식 & 구분 & Panel A 중앙 TV 범위 & Panel B TV 하한 중앙값 범위 & Panel B TV 상한 중앙값 범위 \\",
        r"\midrule",
    ]
    for target in ("exacta", "quinella", "trio"):
        for dimension in DIMENSIONS:
            row = summary[
                summary["target_market"].eq(target) & summary["dimension"].eq(dimension)
            ].iloc[0]
            lines.append(
                f"{target_label[target]} & {dimension_label[dimension]} & "
                f"[{row['panel_a_median_tv_min']:.4f}, {row['panel_a_median_tv_max']:.4f}] & "
                f"[{row['panel_b_median_tv_lower_min']:.4f}, {row['panel_b_median_tv_lower_max']:.4f}] & "
                f"[{row['panel_b_median_tv_upper_min']:.4f}, {row['panel_b_median_tv_upper_max']:.4f}] \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = Path("outputs")
    table_dir = Path("tables")
    bounds = pd.read_csv(output_dir / "main_metrics_bounds.csv")
    panel_a = pd.read_csv(output_dir / "main_heterogeneity.csv")
    panel_b = panel_b_heterogeneity(bounds)
    comparison = comparison_summary(panel_a, panel_b)
    panel_b.to_csv(output_dir / "main_heterogeneity_b.csv", index=False, float_format="%.12g")
    comparison.to_csv(output_dir / "main_heterogeneity_comparison.csv", index=False, float_format="%.12g")
    render_table(comparison, table_dir / "main_heterogeneity_comparison.tex")
    print("PASS: Panel B heterogeneity comparison generated")


if __name__ == "__main__":
    main()
