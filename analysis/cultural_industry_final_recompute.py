#!/usr/bin/env python3
"""Unified manuscript recomputation for the Cultural Industry submission.

The frozen specification is:
- 19,301 complete races, 3,338 clean point races;
- only an odds value equal to 9,999.9 is a censored display-cap observation;
- every reported sampling interval uses a race-date cluster bootstrap with
  99,999 replications, fixed seed 20260816, and percentile 95% limits.

This entry point recomputes manuscript-facing Tables 2--5, the paired TV contrast,
year heterogeneity, and winning-trifecta cap diagnostics from the versioned parsed
odds. Table 4 additionally requires the compact race-market turnover file created by
``python -m analysis.build_turnover_dataset`` from the archived raw KRA pages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow

from analysis.data_audit import read_parquets
from analysis.main_analysis_core import (
    EPSILON,
    SOURCE_MARKET,
    aggregate_point,
    aggregate_price_set,
    harville_trifecta,
    normalize_inverse_odds,
    odds_to_price_set,
    point_metrics,
    point_price_set,
    source_group_index,
    tv_upper_outer,
)
from analysis.main_analysis_fast import tv_lower_exact_fast
from analysis.main_analysis_panels import load_market, validated_realized_index
from analysis.main_analysis_report import crossfitted_calibrated_log_scores
from analysis.main_analysis_runner import common_race_ids, race_metadata, read_frozen_sample


BOOTSTRAP_REPS = 99_999
BOOTSTRAP_SEED = 20260816
BOOTSTRAP_CHUNK = 128
DISPLAY_CAP = 9999.9
CAP_ATOL = 1e-9
TARGETS = ("win", "exacta", "quinella", "trio")
COMPOUND_TARGETS = ("exacta", "quinella", "trio")
CAP_MARKETS = ("win", "exacta", "quinella", "trio", "trifecta")
U_WON = (1_000, 10_000, 50_000, 100_000)
EXPECTED_CAP_RACES = {
    "win": 0,
    "exacta": 581,
    "quinella": 15,
    "trio": 3_434,
    "trifecta": 15_963,
}
MARKET_LABELS = {
    "win": "단승",
    "exacta": "쌍승",
    "quinella": "복승",
    "trio": "삼복승",
    "trifecta": "삼쌍승",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("KRA/parsed"))
    parser.add_argument(
        "--sample-csv", type=Path, default=Path("outputs/analysis_sample.csv")
    )
    parser.add_argument(
        "--turnover-csv",
        type=Path,
        default=Path("data/turnover_by_race_market.csv.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_19301"))
    parser.add_argument(
        "--allow-missing-turnover",
        action="store_true",
        help="finish non-turnover tables while explicitly marking Table 4 unavailable",
    )
    return parser.parse_args()


def canonical_cap(values: Iterable[float]) -> np.ndarray:
    odds = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return np.isclose(odds, DISPLAY_CAP, rtol=0.0, atol=CAP_ATOL)


def canonical_interval(frame: pd.DataFrame):
    odds = pd.to_numeric(frame["odds"], errors="raise").to_numpy(dtype=float)
    return odds_to_price_set(odds, canonical_cap(odds))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ClusterBootstrapPlan:
    dates: tuple[str, ...]
    counts: np.ndarray

    @classmethod
    def build(cls, dates: Iterable[str]) -> "ClusterBootstrapPlan":
        unique = tuple(sorted({str(x) for x in dates}))
        if not unique:
            raise ValueError("bootstrap plan has no race dates")
        g = len(unique)
        if g >= np.iinfo(np.uint16).max:
            raise ValueError("too many clusters for uint16 bootstrap storage")
        counts = np.empty((BOOTSTRAP_REPS, g), dtype=np.uint16)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        probabilities = np.full(g, 1.0 / g, dtype=float)
        generation_chunk = 1024
        for start in range(0, BOOTSTRAP_REPS, generation_chunk):
            end = min(start + generation_chunk, BOOTSTRAP_REPS)
            draw = rng.multinomial(g, probabilities, size=end - start)
            if draw.max(initial=0) >= np.iinfo(np.uint16).max:
                raise OverflowError("cluster count cannot be stored as uint16")
            counts[start:end] = draw.astype(np.uint16)
        return cls(unique, counts)

    @property
    def date_to_code(self) -> dict[str, int]:
        return {date: index for index, date in enumerate(self.dates)}

    def _codes(self, dates: Iterable[str]) -> np.ndarray:
        mapping = self.date_to_code
        result = np.fromiter((mapping[str(x)] for x in dates), dtype=np.int32)
        return result

    def median_draws(self, values: Iterable[float], dates: Iterable[str]) -> np.ndarray:
        values_array = np.asarray(list(values), dtype=float)
        dates_array = np.asarray(list(dates), dtype=object)
        valid = np.isfinite(values_array)
        values_array = values_array[valid]
        dates_array = dates_array[valid]
        if not len(values_array):
            raise ValueError("median bootstrap received no finite values")
        codes = self._codes(dates_array)
        order = np.argsort(values_array, kind="mergesort")
        sorted_values = values_array[order]
        sorted_codes = codes[order]
        draws = np.empty(BOOTSTRAP_REPS, dtype=float)
        for start in range(0, BOOTSTRAP_REPS, BOOTSTRAP_CHUNK):
            end = min(start + BOOTSTRAP_CHUNK, BOOTSTRAP_REPS)
            weights = self.counts[start:end, :][:, sorted_codes]
            cumulative = np.cumsum(weights, axis=1, dtype=np.int32)
            total = cumulative[:, -1]
            if np.any(total <= 0):
                raise RuntimeError("empty resampled subgroup in median bootstrap")
            k1 = (total - 1) // 2
            k2 = total // 2
            i1 = np.argmax(cumulative > k1[:, None], axis=1)
            i2 = np.argmax(cumulative > k2[:, None], axis=1)
            draws[start:end] = (sorted_values[i1] + sorted_values[i2]) / 2.0
        return draws

    def mean_draws(self, values: Iterable[float], dates: Iterable[str]) -> np.ndarray:
        values_array = np.asarray(list(values), dtype=float)
        dates_array = np.asarray(list(dates), dtype=object)
        valid = np.isfinite(values_array)
        values_array = values_array[valid]
        dates_array = dates_array[valid]
        if not len(values_array):
            raise ValueError("mean bootstrap received no finite values")
        codes = self._codes(dates_array)
        g = len(self.dates)
        sums = np.bincount(codes, weights=values_array, minlength=g).astype(float)
        sizes = np.bincount(codes, minlength=g).astype(float)
        draws = np.empty(BOOTSTRAP_REPS, dtype=float)
        for start in range(0, BOOTSTRAP_REPS, 1024):
            end = min(start + 1024, BOOTSTRAP_REPS)
            counts = self.counts[start:end].astype(float, copy=False)
            numerator = counts @ sums
            denominator = counts @ sizes
            if np.any(denominator <= 0):
                raise RuntimeError("empty resampled subgroup in mean bootstrap")
            draws[start:end] = numerator / denominator
        return draws


def percentile_ci(draws: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(draws, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def median_with_ci(
    plan: ClusterBootstrapPlan, values: Iterable[float], dates: Iterable[str]
) -> tuple[float, float, float, np.ndarray]:
    values_array = np.asarray(list(values), dtype=float)
    estimate = float(np.median(values_array[np.isfinite(values_array)]))
    draws = plan.median_draws(values_array, dates)
    low, high = percentile_ci(draws)
    return estimate, low, high, draws


def mean_with_ci(
    plan: ClusterBootstrapPlan, values: Iterable[float], dates: Iterable[str]
) -> tuple[float, float, float, np.ndarray]:
    values_array = np.asarray(list(values), dtype=float)
    estimate = float(np.mean(values_array[np.isfinite(values_array)]))
    draws = plan.mean_draws(values_array, dates)
    low, high = percentile_ci(draws)
    return estimate, low, high, draws


def panel_a_main_harville(
    races: pd.DataFrame,
    trifecta,
    win,
    target_name: str,
    target,
    clean_ids: list[str],
    state_records: list[dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    race_lookup = races.set_index("race_id")
    for race_id in clean_ids:
        source = trifecta.get(race_id)
        actual_frame = target.get(race_id)
        win_frame = win.get(race_id)
        groups = source_group_index(source, actual_frame, target_name)
        cdim = len(actual_frame)
        actual = normalize_inverse_odds(actual_frame["odds"].to_numpy(dtype=float))
        q_main = normalize_inverse_odds(source["odds"].to_numpy(dtype=float))
        q_h = harville_trifecta(source, win_frame)
        main_prediction = aggregate_point(q_main, groups, cdim)
        harville_prediction = aggregate_point(q_h, groups, cdim)
        realized_index, exclusion = validated_realized_index(
            race_lookup.at[race_id, "arrival_tuple"], actual_frame, target_name
        )
        race_date = str(race_lookup.at[race_id, "race_date"])
        year = int(race_lookup.at[race_id, "year"])
        field_size = int(race_lookup.at[race_id, "n_valid_horses"])
        if realized_index is None:
            raise ValueError(f"{race_id}/{target_name}: invalid realized outcome: {exclusion}")

        for model, prediction in (
            ("main", main_prediction),
            ("harville", harville_prediction),
        ):
            rec: dict[str, object] = {
                "race_id": race_id,
                "race_date": race_date,
                "year": year,
                "n_valid_horses": field_size,
                "target_market": target_name,
                "model": model,
            }
            rec.update(point_metrics(actual, prediction))
            realized_probability = float(prediction[realized_index])
            rec["realized_log_score"] = -float(
                np.log(max(realized_probability, EPSILON))
            )
            rec["realized_epsilon_bound"] = realized_probability <= EPSILON
            if model == "main":
                rec["finite_pool_shape"] = float(
                    0.5
                    * math.sqrt(2.0 / math.pi)
                    * np.sqrt(prediction * (1.0 - prediction)).sum()
                )
            else:
                rec["finite_pool_shape"] = float("nan")
            rows.append(rec)
            state_records.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "year": year,
                    "target_market": target_name,
                    "model": model,
                    "predicted": prediction.copy(),
                    "realized_index": realized_index,
                }
            )
    return pd.DataFrame(rows)


def panel_b_main_harville(
    races: pd.DataFrame,
    trifecta,
    win,
    target_name: str,
    target,
    full_ids: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    race_lookup = races.set_index("race_id")
    for race_number, race_id in enumerate(full_ids, 1):
        source = trifecta.get(race_id)
        actual_frame = target.get(race_id)
        win_frame = win.get(race_id)
        groups = source_group_index(source, actual_frame, target_name)
        cdim = len(actual_frame)
        actual_set = canonical_interval(actual_frame)
        source_set = canonical_interval(source)
        main_set = aggregate_price_set(source_set, groups, cdim)
        q_h = harville_trifecta(source, win_frame)
        harville_set = point_price_set(aggregate_point(q_h, groups, cdim))
        race_date = str(race_lookup.at[race_id, "race_date"])
        year = int(race_lookup.at[race_id, "year"])
        field_size = int(race_lookup.at[race_id, "n_valid_horses"])
        for model, prediction_set in (("main", main_set), ("harville", harville_set)):
            lower = tv_lower_exact_fast(actual_set, prediction_set)
            upper = tv_upper_outer(actual_set, prediction_set)
            if lower > upper + 1e-8:
                raise RuntimeError(f"inverted bounds: {race_id}/{target_name}/{model}")
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "year": year,
                    "n_valid_horses": field_size,
                    "target_market": target_name,
                    "model": model,
                    "tv_lower": float(lower),
                    "tv_upper_outer": float(upper),
                }
            )
        if race_number % 2500 == 0:
            print(f"Panel B {target_name}: {race_number:,}/{len(full_ids):,}")
    return pd.DataFrame(rows)


def cap_audit(data_root: Path, scope_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    date_rows: list[dict[str, object]] = []
    for market in CAP_MARKETS:
        frame = read_parquets(data_root, market, columns=["race_id", "odds"])
        frame["race_id"] = frame["race_id"].astype(str)
        frame = frame[frame["race_id"].isin(scope_ids)].copy()
        odds = pd.to_numeric(frame["odds"], errors="raise").to_numpy(dtype=float)
        is_cap = canonical_cap(odds)
        capped_races = int(frame.loc[is_cap, "race_id"].nunique())
        expected = EXPECTED_CAP_RACES[market]
        summary_rows.append(
            {
                "market": market,
                "n_races": int(frame["race_id"].nunique()),
                "capped_races": capped_races,
                "expected_capped_races": expected,
                "count_maintained": capped_races == expected,
                "capped_rows": int(is_cap.sum()),
            }
        )
        target_ids = {rid for rid in scope_ids if rid.startswith("2018-07-01_")}
        date_frame = frame[frame["race_id"].isin(target_ids)]
        date_odds = pd.to_numeric(date_frame["odds"], errors="raise").to_numpy(dtype=float)
        date_rows.append(
            {
                "market": market,
                "date_races": int(date_frame["race_id"].nunique()),
                "rows": len(date_frame),
                "odds_eq_9999_9": int(canonical_cap(date_odds).sum()),
                "odds_gt_9999_9": int((date_odds > DISPLAY_CAP).sum()),
                "max_odds": float(date_odds.max()) if len(date_odds) else float("nan"),
            }
        )
    summary = pd.DataFrame(summary_rows)
    date_check = pd.DataFrame(date_rows)
    if not summary["count_maintained"].all():
        raise AssertionError(f"cap-race count changed:\n{summary.to_string(index=False)}")
    if not date_check["odds_eq_9999_9"].eq(0).all():
        raise AssertionError(f"2018-07-01 contains exact 9999.9:\n{date_check}")
    if not date_check["date_races"].eq(17).all():
        raise AssertionError("2018-07-01 must contain 17 races in every market")
    return summary, date_check


def table3_panel_a(
    metrics: pd.DataFrame, plan: ClusterBootstrapPlan
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for model in ("main", "harville"):
            group = metrics[
                metrics["target_market"].eq(target) & metrics["model"].eq(model)
            ]
            est, lo, hi, _ = median_with_ci(plan, group["tv"], group["race_date"])
            rows.append(
                {
                    "target_market": target,
                    "model": model,
                    "n_races": len(group),
                    "n_race_dates": group["race_date"].nunique(),
                    "median_tv": est,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def paired_tv_differences(
    metrics: pd.DataFrame, plan: ClusterBootstrapPlan
) -> pd.DataFrame:
    pivot = metrics.pivot(
        index=["race_date", "race_id"], columns=["target_market", "model"], values="tv"
    )
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        main = pivot[(target, "main")]
        harville = pivot[(target, "harville")]
        diff = (main - harville).dropna()
        dates = diff.index.get_level_values("race_date")
        est, lo, hi, _ = median_with_ci(plan, diff.to_numpy(), dates)
        rows.append(
            {
                "target_market": target,
                "n_races": len(diff),
                "median_main_minus_harville_tv": est,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


def table3_panel_b(bounds: pd.DataFrame, plan: ClusterBootstrapPlan) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for model in ("main", "harville"):
            group = bounds[
                bounds["target_market"].eq(target) & bounds["model"].eq(model)
            ]
            low_est, low_lo, low_hi, _ = median_with_ci(
                plan, group["tv_lower"], group["race_date"]
            )
            up_est, up_lo, up_hi, _ = median_with_ci(
                plan, group["tv_upper_outer"], group["race_date"]
            )
            rows.append(
                {
                    "target_market": target,
                    "model": model,
                    "n_races": len(group),
                    "median_tv_lower": low_est,
                    "lower_ci_low": low_lo,
                    "lower_ci_high": low_hi,
                    "median_tv_upper": up_est,
                    "upper_ci_low": up_lo,
                    "upper_ci_high": up_hi,
                }
            )
    endpoint = pd.DataFrame(rows)

    superiority_rows: list[dict[str, object]] = []
    for target in COMPOUND_TARGETS:
        target_bounds = bounds[bounds["target_market"].eq(target)]
        lower_h = target_bounds[target_bounds["model"].eq("harville")].set_index("race_id")
        upper_m = target_bounds[target_bounds["model"].eq("main")].set_index("race_id")
        common = lower_h.index.intersection(upper_m.index)
        robust = (
            lower_h.loc[common, "tv_lower"].to_numpy(dtype=float)
            > upper_m.loc[common, "tv_upper_outer"].to_numpy(dtype=float)
        ).astype(float)
        dates = lower_h.loc[common, "race_date"].astype(str).to_numpy()
        est, lo, hi, _ = mean_with_ci(plan, robust, dates)
        superiority_rows.append(
            {
                "target_market": target,
                "n_races": len(common),
                "share_reconstruction_robustly_better": est,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return endpoint, pd.DataFrame(superiority_rows)


def logloss_table(
    metrics: pd.DataFrame,
    state_records: list[dict[str, object]],
    plan: ClusterBootstrapPlan,
) -> pd.DataFrame:
    calibrated = crossfitted_calibrated_log_scores(state_records)
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        raw = metrics[metrics["target_market"].eq(target)].pivot(
            index=["race_date", "race_id"], columns="model", values="realized_log_score"
        )[["main", "harville"]].dropna()
        raw_improvement = (raw["harville"] - raw["main"]).to_numpy(dtype=float)
        dates = raw.index.get_level_values("race_date").to_numpy()
        raw_mean, raw_mean_lo, raw_mean_hi, _ = mean_with_ci(
            plan, raw_improvement, dates
        )
        raw_med, raw_med_lo, raw_med_hi, _ = median_with_ci(
            plan, raw_improvement, dates
        )

        cal = calibrated[calibrated["target_market"].eq(target)].pivot(
            index=["race_date", "race_id"], columns="model", values="calibrated_log_score"
        )[["main", "harville"]].dropna()
        if not cal.index.equals(raw.index):
            raise AssertionError(f"raw/calibrated log-score sample mismatch: {target}")
        cal_improvement = (cal["harville"] - cal["main"]).to_numpy(dtype=float)
        cal_mean, cal_lo, cal_hi, _ = mean_with_ci(plan, cal_improvement, dates)
        rows.append(
            {
                "target_market": target,
                "n_races": len(raw),
                "raw_mean_improvement": raw_mean,
                "raw_mean_ci_low": raw_mean_lo,
                "raw_mean_ci_high": raw_mean_hi,
                "raw_median_improvement": raw_med,
                "raw_median_ci_low": raw_med_lo,
                "raw_median_ci_high": raw_med_hi,
                "calibrated_mean_improvement": cal_mean,
                "calibrated_mean_ci_low": cal_lo,
                "calibrated_mean_ci_high": cal_hi,
            }
        )
    return pd.DataFrame(rows)


def yearly_table(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    yearly_rows: list[dict[str, object]] = []
    for year in sorted(metrics["year"].unique()):
        year_frame = metrics[metrics["year"].eq(year)]
        year_dates = tuple(sorted(year_frame["race_date"].astype(str).unique()))
        plan = ClusterBootstrapPlan.build(year_dates)
        for target in TARGETS:
            for model in ("main", "harville"):
                group = year_frame[
                    year_frame["target_market"].eq(target)
                    & year_frame["model"].eq(model)
                ]
                est, lo, hi, _ = median_with_ci(plan, group["tv"], group["race_date"])
                yearly_rows.append(
                    {
                        "year": int(year),
                        "target_market": target,
                        "model": model,
                        "n_races": len(group),
                        "n_race_dates": len(year_dates),
                        "median_tv": est,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
        del plan
    yearly = pd.DataFrame(yearly_rows)
    ranges = (
        yearly.groupby(["target_market", "model"], as_index=False)["median_tv"]
        .agg(yearly_median_min="min", yearly_median_max="max")
    )
    return yearly, ranges


def winning_trifecta_cap(
    trifecta_frame: pd.DataFrame,
    races: pd.DataFrame,
    full_ids: set[str],
    full_plan: ClusterBootstrapPlan,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = trifecta_frame[trifecta_frame["race_id"].isin(full_ids)].copy()
    odds = pd.to_numeric(frame["odds"], errors="raise").to_numpy(dtype=float)
    frame["canonical_cap"] = canonical_cap(odds)
    hit = frame[frame["is_hit"].fillna(False).astype(bool)].copy()
    hit_counts = hit.groupby("race_id").size()
    if not hit_counts.eq(1).all() or len(hit_counts) != len(full_ids):
        raise AssertionError("each full-sample race must have exactly one winning trifecta row")
    hit = hit.merge(
        races[["race_id", "race_date", "year", "n_valid_horses"]],
        on="race_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_meta"),
    )
    hit["winning_trifecta_capped"] = hit["canonical_cap"].astype(float)
    overall_share, overall_lo, overall_hi, _ = mean_with_ci(
        full_plan, hit["winning_trifecta_capped"], hit["race_date_meta"]
    )
    overall = pd.DataFrame(
        [
            {
                "n_races": len(hit),
                "winning_trifecta_capped_races": int(hit["canonical_cap"].sum()),
                "share": overall_share,
                "ci_low": overall_lo,
                "ci_high": overall_hi,
            }
        ]
    )

    by_year_rows: list[dict[str, object]] = []
    for year, group in hit.groupby("year", sort=True):
        plan = ClusterBootstrapPlan.build(group["race_date_meta"].astype(str).unique())
        share, lo, hi, _ = mean_with_ci(
            plan, group["winning_trifecta_capped"], group["race_date_meta"]
        )
        by_year_rows.append(
            {
                "year": int(year),
                "n_races": len(group),
                "capped_races": int(group["canonical_cap"].sum()),
                "share": share,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
        del plan

    by_field_rows: list[dict[str, object]] = []
    for field_size, group in hit.groupby("n_valid_horses", sort=True):
        share, lo, hi, _ = mean_with_ci(
            full_plan, group["winning_trifecta_capped"], group["race_date_meta"]
        )
        by_field_rows.append(
            {
                "n_valid_horses": int(field_size),
                "n_races": len(group),
                "capped_races": int(group["canonical_cap"].sum()),
                "share": share,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return overall, pd.DataFrame(by_year_rows), pd.DataFrame(by_field_rows)


def load_turnover(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, compression="infer")
    required = {"race_id", "market", "turnover_won"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"turnover file lacks columns: {sorted(missing)}")
    frame["race_id"] = frame["race_id"].astype(str)
    frame["turnover_won"] = pd.to_numeric(frame["turnover_won"], errors="raise")
    if frame.duplicated(["race_id", "market"]).any():
        raise ValueError("turnover file has duplicate race-market rows")
    if (frame["turnover_won"] <= 0).any():
        raise ValueError("turnover must be positive")
    return frame


def turnover_table(
    metrics: pd.DataFrame,
    turnover: pd.DataFrame,
    plan: ClusterBootstrapPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = metrics[
        metrics["target_market"].isin(COMPOUND_TARGETS) & metrics["model"].eq("main")
    ][
        [
            "race_id",
            "race_date",
            "target_market",
            "tv",
            "finite_pool_shape",
        ]
    ].copy()
    wide = turnover.pivot(index="race_id", columns="market", values="turnover_won")
    rows_per_race: list[dict[str, object]] = []
    for target in COMPOUND_TARGETS:
        group = main[main["target_market"].eq(target)].copy()
        if target not in wide.columns or "trifecta" not in wide.columns:
            raise ValueError(f"turnover lacks {target} or trifecta")
        group["target_turnover_won"] = group["race_id"].map(wide[target])
        group["trifecta_turnover_won"] = group["race_id"].map(wide["trifecta"])
        if group[["target_turnover_won", "trifecta_turnover_won"]].isna().any().any():
            missing = group[
                group[["target_turnover_won", "trifecta_turnover_won"]].isna().any(axis=1)
            ]
            raise ValueError(f"{target}: {len(missing)} clean races lack turnover")
        base = group["finite_pool_shape"].to_numpy(dtype=float) * np.sqrt(
            1.0 / group["target_turnover_won"].to_numpy(dtype=float)
            + 1.0 / group["trifecta_turnover_won"].to_numpy(dtype=float)
        )
        observed = group["tv"].to_numpy(dtype=float)
        break_even_u = np.square(observed / base)
        for i, (_, record) in enumerate(group.iterrows()):
            row: dict[str, object] = {
                "race_id": record["race_id"],
                "race_date": record["race_date"],
                "target_market": target,
                "observed_tv": observed[i],
                "target_turnover_won": float(record["target_turnover_won"]),
                "trifecta_turnover_won": float(record["trifecta_turnover_won"]),
                "noise_base_per_sqrt_won": base[i],
                "break_even_u_won": break_even_u[i],
            }
            for u in U_WON:
                row[f"noise_tv_u_{u}"] = base[i] * math.sqrt(u)
            rows_per_race.append(row)
    per_race = pd.DataFrame(rows_per_race)

    summary_rows: list[dict[str, object]] = []
    for target in COMPOUND_TARGETS:
        group = per_race[per_race["target_market"].eq(target)]
        observed_est, observed_lo, observed_hi, observed_draws = median_with_ci(
            plan, group["observed_tv"], group["race_date"]
        )
        break_est, break_lo, break_hi, _ = median_with_ci(
            plan, group["break_even_u_won"], group["race_date"]
        )
        for u in U_WON:
            col = f"noise_tv_u_{u}"
            noise_est, noise_lo, noise_hi, noise_draws = median_with_ci(
                plan, group[col], group["race_date"]
            )
            ratio = 100.0 * noise_est / observed_est
            ratio_draws = 100.0 * noise_draws / observed_draws
            ratio_lo, ratio_hi = percentile_ci(ratio_draws)
            summary_rows.append(
                {
                    "target_market": target,
                    "u_won": u,
                    "n_races": len(group),
                    "observed_median_tv": observed_est,
                    "observed_ci_low": observed_lo,
                    "observed_ci_high": observed_hi,
                    "expected_noise_median_tv": noise_est,
                    "noise_ci_low": noise_lo,
                    "noise_ci_high": noise_hi,
                    "noise_to_observed_percent": ratio,
                    "ratio_ci_low": ratio_lo,
                    "ratio_ci_high": ratio_hi,
                    "break_even_u_median_won": break_est,
                    "break_even_u_ci_low": break_lo,
                    "break_even_u_ci_high": break_hi,
                }
            )
    return pd.DataFrame(summary_rows), per_race


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def write_markdown(
    path: Path,
    cap_summary: pd.DataFrame,
    date_check: pd.DataFrame,
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    superiority: pd.DataFrame,
    paired: pd.DataFrame,
    turnover: pd.DataFrame | None,
    logloss: pd.DataFrame,
    winning: pd.DataFrame,
    yearly_ranges: pd.DataFrame,
) -> None:
    def f(x: float, digits: int = 6) -> str:
        return f"{float(x):.{digits}f}"

    lines = [
        "# 문화산업연구 최종 표본 일괄 재산출",
        "",
        f"- 전체표본: **19,301경주**",
        f"- 점표본: **3,338경주 (979 경주일)**",
        f"- bootstrap: 경주일 군집 **{BOOTSTRAP_REPS:,}회**, seed **{BOOTSTRAP_SEED}**, percentile 95% CI",
        "- 표시상한: **정확히 9,999.9인 게시값만** 검열값으로 처리",
        "",
        "## 사전 확인 — 2018-07-01 및 승식별 표시상한",
        "",
        cap_summary.to_markdown(index=False),
        "",
        date_check.to_markdown(index=False),
        "",
        "## 표 3 Panel A — 점표본",
        "",
        panel_a.to_markdown(index=False),
        "",
        "## 표 3 Panel B — 전체표본 부분식별",
        "",
        panel_b.to_markdown(index=False),
        "",
        "### 재구성의 강건한 우위 경주 비율",
        "",
        superiority.to_markdown(index=False),
        "",
        "## 짝지은 TV 차이: 재구성 - Harville",
        "",
        paired.to_markdown(index=False),
        "",
        "## 표 4 — 실제 매출액 기반 유한풀 잡음",
        "",
    ]
    if turnover is None:
        lines += [
            "**미산출:** compact `turnover_by_race_market.csv.gz`가 아직 체크인되지 않아 Table 4를 임의의 값으로 채우지 않았다.",
            "",
        ]
    else:
        lines += [turnover.to_markdown(index=False), ""]
    lines += [
        "## 표 5 — 로그손실 개선 (Harville - 재구성)",
        "",
        logloss.to_markdown(index=False),
        "",
        "## 당첨 삼쌍승 조합의 9,999.9 표시",
        "",
        winning.to_markdown(index=False),
        "",
        "## Panel A 연도별 중앙값 범위",
        "",
        yearly_ranges.to_markdown(index=False),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = read_frozen_sample(args.sample_csv)
    clean_ids = common_race_ids(sample, "eligible_clean_point_sample")
    full_ids = common_race_ids(sample, "eligible_complete_sample")
    if len(clean_ids) != 3338 or len(full_ids) != 19301:
        raise AssertionError(
            f"frozen sample mismatch: clean={len(clean_ids)}, full={len(full_ids)}"
        )

    races = race_metadata(args.data_root, set(full_ids))
    clean_dates = tuple(
        sorted(races[races["race_id"].isin(clean_ids)]["race_date"].astype(str).unique())
    )
    full_dates = tuple(sorted(races["race_date"].astype(str).unique()))
    if len(clean_dates) != 979:
        raise AssertionError(f"clean sample must contain 979 race dates, got {len(clean_dates)}")
    print(f"samples: full={len(full_ids):,}, clean={len(clean_ids):,}, clean dates={len(clean_dates):,}, full dates={len(full_dates):,}")

    cap_summary, date_check = cap_audit(args.data_root, set(full_ids))
    write_csv(cap_summary, args.output_dir / "table2_cap_counts.csv")
    write_csv(date_check, args.output_dir / "precheck_2018_07_01.csv")

    trifecta = load_market(args.data_root, SOURCE_MARKET, set(full_ids))
    win = load_market(args.data_root, "win", set(full_ids))
    metrics_frames: list[pd.DataFrame] = []
    bounds_frames: list[pd.DataFrame] = []
    state_records: list[dict[str, object]] = []
    for target_name in TARGETS:
        target = load_market(args.data_root, target_name, set(full_ids))
        metrics_frames.append(
            panel_a_main_harville(
                races, trifecta, win, target_name, target, clean_ids, state_records
            )
        )
        bounds_frames.append(
            panel_b_main_harville(
                races, trifecta, win, target_name, target, full_ids
            )
        )
    metrics = pd.concat(metrics_frames, ignore_index=True)
    bounds = pd.concat(bounds_frames, ignore_index=True)
    write_csv(metrics, args.output_dir / "panel_a_per_race_main_harville.csv")
    write_csv(bounds, args.output_dir / "panel_b_per_race_main_harville.csv")

    clean_plan = ClusterBootstrapPlan.build(clean_dates)
    full_plan = clean_plan if clean_dates == full_dates else ClusterBootstrapPlan.build(full_dates)

    panel_a_summary = table3_panel_a(metrics, clean_plan)
    paired = paired_tv_differences(metrics, clean_plan)
    panel_b_summary, superiority = table3_panel_b(bounds, full_plan)
    logloss = logloss_table(metrics, state_records, clean_plan)
    yearly, yearly_ranges = yearly_table(metrics)
    winning, winning_by_year, winning_by_field = winning_trifecta_cap(
        trifecta.frame, races, set(full_ids), full_plan
    )

    write_csv(panel_a_summary, args.output_dir / "table3_panel_a.csv")
    write_csv(panel_b_summary, args.output_dir / "table3_panel_b.csv")
    write_csv(superiority, args.output_dir / "table3_panel_b_superiority.csv")
    write_csv(paired, args.output_dir / "paired_tv_difference.csv")
    write_csv(logloss, args.output_dir / "table5_logloss.csv")
    write_csv(yearly, args.output_dir / "panel_a_by_year.csv")
    write_csv(yearly_ranges, args.output_dir / "panel_a_yearly_ranges.csv")
    write_csv(winning, args.output_dir / "winning_trifecta_cap_overall.csv")
    write_csv(winning_by_year, args.output_dir / "winning_trifecta_cap_by_year.csv")
    write_csv(winning_by_field, args.output_dir / "winning_trifecta_cap_by_field.csv")

    turnover_frame = load_turnover(args.turnover_csv)
    turnover_summary: pd.DataFrame | None = None
    if turnover_frame is None:
        status = pd.DataFrame(
            [
                {
                    "available": False,
                    "required_file": args.turnover_csv.as_posix(),
                    "reason": "compact turnover file not found",
                }
            ]
        )
        write_csv(status, args.output_dir / "table4_status.csv")
        if not args.allow_missing_turnover:
            raise FileNotFoundError(
                f"{args.turnover_csv} is required for Table 4; build it with "
                "python -m analysis.build_turnover_dataset --raw-root <Dropbox raw root>"
            )
    else:
        turnover_summary, turnover_per_race = turnover_table(
            metrics, turnover_frame, clean_plan
        )
        write_csv(turnover_summary, args.output_dir / "table4_turnover_noise.csv")
        write_csv(turnover_per_race, args.output_dir / "table4_turnover_noise_per_race.csv")
        write_csv(
            pd.DataFrame([{"available": True, "turnover_sha256": file_sha256(args.turnover_csv)}]),
            args.output_dir / "table4_status.csv",
        )

    report_path = args.output_dir / "MANUSCRIPT_REPLACEMENT_TABLES.md"
    write_markdown(
        report_path,
        cap_summary,
        date_check,
        panel_a_summary,
        panel_b_summary,
        superiority,
        paired,
        turnover_summary,
        logloss,
        winning,
        yearly_ranges,
    )

    output_files = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    manifest = {
        "sample": {"full_races": len(full_ids), "clean_races": len(clean_ids), "clean_race_dates": len(clean_dates)},
        "cap_definition": "odds == 9999.9 within absolute tolerance 1e-9; odds > 9999.9 are point observations",
        "bootstrap": {
            "unit": "race_date cluster",
            "reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95%",
            "common_resamples_within_date_universe": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "git_sha": os.environ.get("GITHUB_SHA", ""),
        "turnover_available": turnover_frame is not None,
        "outputs": {
            p.name: {"size_bytes": p.stat().st_size, "sha256": file_sha256(p)}
            for p in output_files
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(panel_a_summary.to_string(index=False))
    print(panel_b_summary.to_string(index=False))
    print(superiority.to_string(index=False))
    print(paired.to_string(index=False))
    print(logloss.to_string(index=False))
    print(winning.to_string(index=False))
    if turnover_summary is not None:
        print(turnover_summary.to_string(index=False))
    print(f"PASS: unified final recomputation -> {args.output_dir}")


if __name__ == "__main__":
    main()
