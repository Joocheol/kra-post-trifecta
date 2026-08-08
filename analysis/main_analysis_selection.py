"""Render frozen clean-versus-capped selection diagnostics for the manuscript."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _share(composition: pd.DataFrame, group: str, dimension: str, level: str) -> float:
    rows = composition[
        composition["sample_group"].eq(group)
        & composition["dimension"].eq(dimension)
        & composition["level"].astype(str).eq(str(level))
    ]
    if len(rows) != 1:
        raise ValueError(f"missing unique composition cell: {group}/{dimension}/{level}")
    return float(rows.iloc[0]["share"])


def _period_share(composition: pd.DataFrame, group: str, years: set[int]) -> float:
    rows = composition[
        composition["sample_group"].eq(group)
        & composition["dimension"].eq("year")
    ].copy()
    rows["year"] = rows["level"].astype(int)
    return float(rows.loc[rows["year"].isin(years), "share"].sum())


def render_table(
    selection: pd.DataFrame,
    composition: pd.DataFrame,
    tails: pd.DataFrame,
    output: Path,
) -> None:
    selection = selection.set_index("sample_group")
    tails = tails.set_index("sample_group")
    expected = {"clean", "capped"}
    if set(selection.index) != expected or set(tails.index) != expected:
        raise ValueError("selection diagnostics require clean and capped groups")

    labels = {"clean": "clean", "capped": "상한 포함"}
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"표본 & 경주 수 & 출전두수 중앙값 [IQR] & 2016--19 비중 & meet 1 & meet 2 & meet 3 & 비검열 배당 95\% & 99\% \\",
        r"\midrule",
    ]
    for group in ("clean", "capped"):
        row = selection.loc[group]
        tail = tails.loc[group]
        field = (
            f"{int(row['field_size_median'])} "
            f"[{int(row['field_size_q25'])}, {int(row['field_size_q75'])}]"
        )
        early = _period_share(composition, group, {2016, 2017, 2018, 2019})
        m1 = _share(composition, group, "meet", "1")
        m2 = _share(composition, group, "meet", "2")
        m3 = _share(composition, group, "meet", "3")
        lines.append(
            f"{labels[group]} & {int(row['n_races']):,} & {field} & "
            f"{100*early:.1f}\% & {100*m1:.1f}\% & {100*m2:.1f}\% & {100*m3:.1f}\% & "
            f"{float(tail['odds_q95']):,.1f} & {float(tail['odds_q99']):,.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def heterogeneity_summary(heterogeneity: pd.DataFrame) -> pd.DataFrame:
    main = heterogeneity[heterogeneity["model"].eq("main")]
    summary = (
        main.groupby(["target_market", "dimension"], sort=True)["median_tv"]
        .agg(n_levels="size", min_group_median_tv="min", max_group_median_tv="max")
        .reset_index()
    )
    return summary


def main() -> None:
    output_dir = Path("outputs")
    table_dir = Path("tables")
    selection = pd.read_csv(output_dir / "main_sample_selection.csv")
    composition = pd.read_csv(output_dir / "main_sample_composition.csv")
    tails = pd.read_csv(output_dir / "main_sample_tail.csv")
    heterogeneity = pd.read_csv(output_dir / "main_heterogeneity.csv")
    render_table(selection, composition, tails, table_dir / "main_sample_selection.tex")
    heterogeneity_summary(heterogeneity).to_csv(
        output_dir / "main_heterogeneity_summary.csv", index=False, float_format="%.12g"
    )
    print("PASS: sample-selection and compact heterogeneity diagnostics generated")


if __name__ == "__main__":
    main()
