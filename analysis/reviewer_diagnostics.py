#!/usr/bin/env python3
"""Reviewer-requested diagnostics for odds levels and finite-pool precision.

Archived KRA pages contain race-market turnover, and the revised raw-data parser
preserves it. Total turnover alone does not identify the number of independent
betting decisions or the effective ticket count. The finite-pool calculation is
therefore an explicitly labelled effective-ticket sensitivity analysis, not an
estimate of the actual sampling-noise floor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data_audit import read_parquets, prepare_races


MARKETS = ("win", "exacta", "quinella", "trio", "trifecta")
TARGET_MARKETS = ("exacta", "quinella", "trio")
EFFECTIVE_TICKET_GRID = (1_000, 10_000, 100_000, 1_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("KRA/parsed"))
    parser.add_argument(
        "--sample-csv", type=Path, default=Path("outputs/analysis_sample.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))
    return parser.parse_args()


def in_scope_ids(data_root: Path) -> set[str]:
    races = prepare_races(data_root)
    return set(races.loc[races["in_date_scope"], "race_id"].astype(str))


def odds_level_frame(data_root: Path, race_ids: set[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for market in MARKETS:
        frame = read_parquets(
            data_root, market, columns=["race_id", "odds", "is_capped_odds"]
        )
        frame = frame[frame["race_id"].isin(race_ids)].copy()
        frame["inverse_odds"] = 1.0 / frame["odds"].astype(float)
        grouped = (
            frame.groupby("race_id", as_index=False)
            .agg(
                sum_inverse_odds=("inverse_odds", "sum"),
                n_combinations=("odds", "size"),
                n_capped_rows=("is_capped_odds", "sum"),
                any_capped=("is_capped_odds", "any"),
            )
        )
        grouped["market"] = market
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def odds_level_summary(levels: pd.DataFrame) -> pd.DataFrame:
    return (
        levels.groupby(["market", "any_capped"], as_index=False)
        .agg(
            n_races=("race_id", "nunique"),
            median_sum_inverse_odds=("sum_inverse_odds", "median"),
            p05_sum_inverse_odds=("sum_inverse_odds", lambda x: x.quantile(0.05)),
            p95_sum_inverse_odds=("sum_inverse_odds", lambda x: x.quantile(0.95)),
            median_combinations=("n_combinations", "median"),
            median_capped_rows=("n_capped_rows", "median"),
        )
        .sort_values(["market", "any_capped"])
        .reset_index(drop=True)
    )


def display_cap_summary(levels: pd.DataFrame) -> pd.DataFrame:
    return (
        levels.groupby("market", as_index=False)
        .agg(
            n_races=("race_id", "nunique"),
            capped_races=("any_capped", "sum"),
            capped_rows=("n_capped_rows", "sum"),
        )
        .assign(capped_race_share=lambda x: x["capped_races"] / x["n_races"])
        .sort_values("market")
        .reset_index(drop=True)
    )


def clean_ids(sample_csv: Path, market: str) -> set[str]:
    sample = pd.read_csv(sample_csv, low_memory=False)
    required = {"race_id", "target_market", "eligible_clean_point_sample"}
    missing = required.difference(sample.columns)
    if missing:
        raise ValueError(f"analysis sample lacks columns: {sorted(missing)}")
    flag = sample["eligible_clean_point_sample"]
    if not pd.api.types.is_bool_dtype(flag):
        flag = flag.astype(str).str.lower().map({"true": True, "false": False})
    return set(
        sample.loc[sample["target_market"].eq(market) & flag.fillna(False), "race_id"]
        .astype(str)
    )


def expected_two_pool_tv(probability: np.ndarray, effective_n: int) -> float:
    """Normal-approximation TV for two independent equal-size multinomial pools."""
    probability = np.asarray(probability, dtype=float)
    variance = probability * (1.0 - probability) * (2.0 / effective_n)
    return float(0.5 * np.sqrt(2.0 / np.pi) * np.sqrt(variance).sum())


def finite_pool_reference(
    data_root: Path,
    sample_csv: Path,
    observed_summary_csv: Path,
) -> pd.DataFrame:
    observed = pd.read_csv(observed_summary_csv)
    observed = observed[observed["model"].eq("main")].set_index("target_market")
    rows: list[dict[str, object]] = []
    for market in TARGET_MARKETS:
        ids = clean_ids(sample_csv, market)
        frame = read_parquets(data_root, market, columns=["race_id", "odds"])
        frame = frame[frame["race_id"].isin(ids)].copy()
        frame["inverse_odds"] = 1.0 / frame["odds"].astype(float)
        frame["price_share"] = frame["inverse_odds"] / frame.groupby("race_id")[
            "inverse_odds"
        ].transform("sum")
        grouped = list(frame.groupby("race_id", sort=False)["price_share"])
        for effective_n in EFFECTIVE_TICKET_GRID:
            reference = np.asarray(
                [
                    expected_two_pool_tv(values.to_numpy(), effective_n)
                    for _, values in grouped
                ]
            )
            rows.append(
                {
                    "target_market": market,
                    "effective_tickets_per_pool": effective_n,
                    "n_races": len(reference),
                    "median_reference_tv": float(np.median(reference)),
                    "p05_reference_tv": float(np.quantile(reference, 0.05)),
                    "p95_reference_tv": float(np.quantile(reference, 0.95)),
                    "observed_main_median_tv": float(observed.loc[market, "median_tv"]),
                    "observed_to_reference_ratio": float(
                        observed.loc[market, "median_tv"] / np.median(reference)
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    labels = {"exacta": "쌍승", "quinella": "복승", "trio": "삼복승"}
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"승식 & 풀당 유효표 수 & 기준 TV 중앙값 & 관측 TV & 관측/기준 \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{labels[row['target_market']]} & {int(row['effective_tickets_per_pool']):,} & "
            f"{row['median_reference_tv']:.4f} & {row['observed_main_median_tv']:.4f} & "
            f"{row['observed_to_reference_ratio']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.sample_csv.exists():
        raise FileNotFoundError(
            f"{args.sample_csv} is missing; run `python -m analysis.data_audit --strict`"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = in_scope_ids(args.data_root)
    levels = odds_level_frame(args.data_root, ids)
    overround = odds_level_summary(levels)
    caps = display_cap_summary(levels)
    finite = finite_pool_reference(
        args.data_root,
        args.sample_csv,
        args.output_dir / "main_panel_a_summary.csv",
    )
    overround.to_csv(args.output_dir / "odds_overround_summary.csv", index=False)
    caps.to_csv(args.output_dir / "display_cap_summary.csv", index=False)
    finite.to_csv(args.output_dir / "finite_pool_reference.csv", index=False)
    write_latex_table(finite, args.table_dir / "finite_pool_reference.tex")
    print(overround.to_string(index=False))
    print(finite.to_string(index=False))


if __name__ == "__main__":
    main()
