#!/usr/bin/env python3
"""Reviewer-requested diagnostics for odds levels and finite-pool precision.

Archived KRA pages contain race-market turnover, and the revised raw-data parser
preserves it. Total turnover alone does not identify the number of independent
betting decisions or the effective ticket count. We therefore report two distinct
finite-pool diagnostics:

1. an effective-ticket sensitivity analysis over a transparent grid; and
2. a turnover-matched *mechanical* null in which each 100 won wagering unit is
   treated as an independent multinomial draw from one common latent price vector.

The second calculation directly answers the referee's proposed benchmark, but its
independence assumption is deliberately labelled as strong. It is not an estimate of
the actual number of independent bettor decisions.
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
NOMINAL_WAGER_UNIT_WON = 100
TURNOVER_SIM_DRAWS = 64
TURNOVER_SIM_SEED = 20260814
TARGET_KEYS = {
    "exacta": ("first_no", "second_no"),
    "quinella": ("horse_a", "horse_b"),
    "trio": ("horse_a", "horse_b", "horse_c"),
}


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


def _normalised_inverse_odds(frame: pd.DataFrame) -> pd.Series:
    inverse = 1.0 / pd.to_numeric(frame["odds"], errors="raise").astype(float)
    total = float(inverse.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("invalid inverse-odds total")
    return inverse / total


def _target_price_series(frame: pd.DataFrame, market: str) -> pd.Series:
    keys = list(TARGET_KEYS[market])
    local = frame.copy()
    local["price_share"] = _normalised_inverse_odds(local).to_numpy()
    result = local.set_index(keys)["price_share"].sort_index()
    if result.index.has_duplicates:
        raise ValueError(f"duplicate {market} support in observed target prices")
    return result


def _trifecta_marginal_series(frame: pd.DataFrame, market: str) -> pd.Series:
    local = frame.copy()
    local["price_share"] = _normalised_inverse_odds(local).to_numpy()
    if market == "exacta":
        result = local.groupby(["first_no", "second_no"], sort=True)["price_share"].sum()
    elif market == "quinella":
        first = local["first_no"].to_numpy(dtype=np.int64)
        second = local["second_no"].to_numpy(dtype=np.int64)
        local["horse_a"] = np.minimum(first, second)
        local["horse_b"] = np.maximum(first, second)
        result = local.groupby(["horse_a", "horse_b"], sort=True)["price_share"].sum()
    elif market == "trio":
        ordered = np.sort(
            local[["first_no", "second_no", "third_no"]].to_numpy(dtype=np.int64),
            axis=1,
        )
        local[["horse_a", "horse_b", "horse_c"]] = ordered
        result = local.groupby(
            ["horse_a", "horse_b", "horse_c"], sort=True
        )["price_share"].sum()
    else:
        raise ValueError(f"unsupported target market: {market}")
    return result.sort_index()


def _turnover_frame(data_root: Path, race_ids: set[str]) -> pd.DataFrame:
    status = read_parquets(
        data_root,
        "market_status",
        columns=["race_id", "market", "turnover_won"],
    )
    status = status[
        status["race_id"].astype(str).isin(race_ids)
        & status["market"].isin((*TARGET_MARKETS, "trifecta"))
    ].copy()
    duplicated = status.duplicated(["race_id", "market"], keep=False)
    if duplicated.any():
        raise ValueError("duplicated race-market turnover rows")
    wide = status.pivot(index="race_id", columns="market", values="turnover_won")
    return wide


def turnover_matched_finite_pool(
    data_root: Path,
    sample_csv: Path,
) -> pd.DataFrame:
    """Simulate the referee's same-latent-price, turnover-matched null.

    Each observed 100 won of turnover is treated as one independent multinomial
    trial.  At target-event level, marginalising a multinomial trifecta sample is
    itself multinomial, so it is exactly equivalent (and much cheaper) to draw two
    target-support multinomials with the target and trifecta nominal sample sizes.
    """
    rng = np.random.default_rng(TURNOVER_SIM_SEED)
    rows: list[dict[str, object]] = []

    trifecta_columns = [
        "race_id",
        "first_no",
        "second_no",
        "third_no",
        "odds",
    ]
    all_clean = set().union(*(clean_ids(sample_csv, m) for m in TARGET_MARKETS))
    trifecta = read_parquets(data_root, "trifecta", columns=trifecta_columns)
    trifecta = trifecta[trifecta["race_id"].astype(str).isin(all_clean)].copy()
    try:
        turnover = _turnover_frame(data_root, all_clean)
    except Exception as exc:
        if "turnover_won" in str(exc):
            return pd.DataFrame()
        raise

    for market in TARGET_MARKETS:
        market_ids = clean_ids(sample_csv, market)
        target = read_parquets(
            data_root,
            market,
            columns=["race_id", *TARGET_KEYS[market], "odds"],
        )
        target = target[target["race_id"].astype(str).isin(market_ids)].copy()
        target_groups = {str(k): v for k, v in target.groupby("race_id", sort=False)}
        trifecta_groups = {
            str(k): v for k, v in trifecta[trifecta["race_id"].astype(str).isin(market_ids)]
            .groupby("race_id", sort=False)
        }

        race_rows: list[dict[str, float | str]] = []
        for race_id in sorted(market_ids):
            if race_id not in target_groups or race_id not in trifecta_groups:
                continue
            if race_id not in turnover.index:
                continue
            target_turnover = turnover.at[race_id, market] if market in turnover.columns else np.nan
            trifecta_turnover = (
                turnover.at[race_id, "trifecta"] if "trifecta" in turnover.columns else np.nan
            )
            if pd.isna(target_turnover) or pd.isna(trifecta_turnover):
                continue
            target_turnover = int(target_turnover)
            trifecta_turnover = int(trifecta_turnover)
            if target_turnover <= 0 or trifecta_turnover <= 0:
                continue
            if target_turnover % NOMINAL_WAGER_UNIT_WON != 0:
                raise ValueError(f"target turnover is not on 100-won grid: {race_id} {market}")
            if trifecta_turnover % NOMINAL_WAGER_UNIT_WON != 0:
                raise ValueError(f"trifecta turnover is not on 100-won grid: {race_id}")

            observed = _target_price_series(target_groups[race_id], market)
            latent = _trifecta_marginal_series(trifecta_groups[race_id], market)
            if not observed.index.equals(latent.index):
                raise ValueError(f"support mismatch in turnover benchmark: {race_id} {market}")
            p = latent.to_numpy(dtype=float)
            p = p / p.sum()
            observed_tv = float(
                0.5 * np.abs(observed.to_numpy(dtype=float) - p).sum()
            )
            n_target = target_turnover // NOMINAL_WAGER_UNIT_WON
            n_trifecta = trifecta_turnover // NOMINAL_WAGER_UNIT_WON
            target_draws = rng.multinomial(n_target, p, size=TURNOVER_SIM_DRAWS) / n_target
            trifecta_draws = (
                rng.multinomial(n_trifecta, p, size=TURNOVER_SIM_DRAWS) / n_trifecta
            )
            simulated_tv = 0.5 * np.abs(target_draws - trifecta_draws).sum(axis=1)
            null_median = float(np.median(simulated_tv))
            null_p95 = float(np.quantile(simulated_tv, 0.95))
            race_rows.append(
                {
                    "race_id": race_id,
                    "target_turnover_won": float(target_turnover),
                    "trifecta_turnover_won": float(trifecta_turnover),
                    "target_nominal_units": float(n_target),
                    "trifecta_nominal_units": float(n_trifecta),
                    "observed_tv": observed_tv,
                    "null_median_tv": null_median,
                    "null_p95_tv": null_p95,
                    "observed_gt_null_p95": float(observed_tv > null_p95),
                }
            )

        race_frame = pd.DataFrame(race_rows)
        if race_frame.empty:
            raise ValueError(f"no turnover-complete clean races for {market}")
        null_median = float(race_frame["null_median_tv"].median())
        observed_median = float(race_frame["observed_tv"].median())
        rows.append(
            {
                "target_market": market,
                "n_clean_races": len(market_ids),
                "n_turnover_complete_races": len(race_frame),
                "simulation_draws_per_race": TURNOVER_SIM_DRAWS,
                "nominal_unit_won": NOMINAL_WAGER_UNIT_WON,
                "median_target_turnover_won": float(
                    race_frame["target_turnover_won"].median()
                ),
                "median_trifecta_turnover_won": float(
                    race_frame["trifecta_turnover_won"].median()
                ),
                "median_target_nominal_units": float(
                    race_frame["target_nominal_units"].median()
                ),
                "median_trifecta_nominal_units": float(
                    race_frame["trifecta_nominal_units"].median()
                ),
                "observed_median_tv": observed_median,
                "median_null_tv": null_median,
                "p05_race_null_median_tv": float(
                    race_frame["null_median_tv"].quantile(0.05)
                ),
                "p95_race_null_median_tv": float(
                    race_frame["null_median_tv"].quantile(0.95)
                ),
                "median_race_null_p95_tv": float(race_frame["null_p95_tv"].median()),
                "observed_to_null_median_ratio": observed_median / null_median,
                "share_observed_gt_race_null_p95": float(
                    race_frame["observed_gt_null_p95"].mean()
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
    turnover_null = turnover_matched_finite_pool(args.data_root, args.sample_csv)
    overround.to_csv(args.output_dir / "odds_overround_summary.csv", index=False)
    caps.to_csv(args.output_dir / "display_cap_summary.csv", index=False)
    finite.to_csv(args.output_dir / "finite_pool_reference.csv", index=False)
    turnover_null.to_csv(
        args.output_dir / "turnover_matched_finite_pool.csv", index=False
    )
    write_latex_table(finite, args.table_dir / "finite_pool_reference.tex")
    print(overround.to_string(index=False))
    print(finite.to_string(index=False))
    print("Turnover-matched mechanical null (100-won units treated as independent):")
    print(turnover_null.to_string(index=False))


if __name__ == "__main__":
    main()
