"""Descriptive price-coherence comparison around Korea's online wagering launch.

This is a reviewer diagnostic, not a causal design.  It keeps the frozen race
sample and the existing main-analysis definitions, restricts evaluation to 2024,
and compares races before vs. on/after 2024-06-21.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data_audit import TARGET_MARKETS
from analysis.main_analysis_core import SOURCE_MARKET, stable_uint
from analysis.main_analysis_guards import assert_win_uncapped
from analysis.main_analysis_panels import grouped_ids_by_field, load_market, panel_a, panel_b
from analysis.main_analysis_runner import common_race_ids, race_metadata, read_frozen_sample

LAUNCH_DATE = pd.Timestamp("2024-06-21")
MODELS = {"main", "harville"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("KRA/parsed"))
    p.add_argument("--sample-csv", type=Path, default=Path("outputs/analysis_sample.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    return p.parse_args()


def median_ci(values: np.ndarray, *, label: str, reps: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.median(x))
    if len(x) == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(stable_uint(f"online-launch|{label}"))
    draws = np.empty(reps, dtype=float)
    for b in range(reps):
        draws[b] = np.median(x[rng.integers(0, len(x), size=len(x))])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return estimate, float(lo), float(hi)


def summarize_a(metrics: pd.DataFrame, reps: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, target, model), g in metrics.groupby(["period", "target_market", "model"], sort=True):
        med, lo, hi = median_ci(g["tv"].to_numpy(), label=f"A|{period}|{target}|{model}", reps=reps)
        rows.append({
            "panel": "A",
            "period": period,
            "target_market": target,
            "model": model,
            "n_races": int(g["race_id"].nunique()),
            "median_tv": med,
            "median_tv_ci_low": lo,
            "median_tv_ci_high": hi,
            "q25_tv": float(g["tv"].quantile(0.25)),
            "q75_tv": float(g["tv"].quantile(0.75)),
        })
    return pd.DataFrame(rows)


def summarize_b(bounds: pd.DataFrame, reps: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, target, model), g in bounds.groupby(["period", "target_market", "model"], sort=True):
        low_med, low_lo, low_hi = median_ci(g["tv_lower"].to_numpy(), label=f"B-low|{period}|{target}|{model}", reps=reps)
        up_med, up_lo, up_hi = median_ci(g["tv_upper_outer"].to_numpy(), label=f"B-up|{period}|{target}|{model}", reps=reps)
        rows.append({
            "panel": "B",
            "period": period,
            "target_market": target,
            "model": model,
            "n_races": int(g["race_id"].nunique()),
            "median_tv_lower": low_med,
            "median_tv_lower_ci_low": low_lo,
            "median_tv_lower_ci_high": low_hi,
            "median_tv_upper_outer": up_med,
            "median_tv_upper_outer_ci_low": up_lo,
            "median_tv_upper_outer_ci_high": up_hi,
        })
    return pd.DataFrame(rows)


def add_period(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    dates = metadata[["race_id", "race_date"]].copy()
    dates["race_date"] = pd.to_datetime(dates["race_date"])
    out = frame.merge(dates, on="race_id", how="left", validate="many_to_one")
    if out["race_date"].isna().any():
        raise ValueError("missing race dates in online-launch diagnostic")
    out["period"] = np.where(out["race_date"] < LAUNCH_DATE, "pre", "post")
    return out


def main() -> None:
    args = parse_args()
    sample = read_frozen_sample(args.sample_csv)
    all_clean_ids = common_race_ids(sample, "eligible_clean_point_sample")
    all_full_ids = common_race_ids(sample, "eligible_complete_sample")
    source_ids = set(all_full_ids)
    races = race_metadata(args.data_root, source_ids)
    races["race_date"] = pd.to_datetime(races["race_date"])
    races_2024 = races[races["race_date"].dt.year.eq(2024)].copy()
    clean_2024 = sorted(set(all_clean_ids).intersection(races_2024["race_id"]))
    full_2024 = sorted(set(all_full_ids).intersection(races_2024["race_id"]))
    if not clean_2024 or not full_2024:
        raise ValueError("2024 evaluation sample is empty")

    trifecta = load_market(args.data_root, SOURCE_MARKET, source_ids)
    win = load_market(args.data_root, "win", source_ids)
    assert_win_uncapped(win.frame)
    clean_peers = grouped_ids_by_field(races, all_clean_ids)
    full_peers = grouped_ids_by_field(races, all_full_ids)

    a_frames: list[pd.DataFrame] = []
    b_frames: list[pd.DataFrame] = []
    state_records: list[dict[str, object]] = []
    for target_name in TARGET_MARKETS:
        target = load_market(args.data_root, target_name, source_ids)
        a = panel_a(races, trifecta, win, target_name, target, clean_2024, clean_peers, state_records)
        a_frames.append(a[a["model"].isin(MODELS)].copy())
        b = panel_b(races, trifecta, win, target_name, target, full_2024, full_peers)
        b_frames.append(b[b["model"].isin(MODELS)].copy())

    metrics = add_period(pd.concat(a_frames, ignore_index=True), races_2024)
    bounds = add_period(pd.concat(b_frames, ignore_index=True), races_2024)
    summary_a = summarize_a(metrics, args.bootstrap_reps)
    summary_b = summarize_b(bounds, args.bootstrap_reps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_a.to_csv(args.output_dir / "online_launch_panel_a.csv", index=False, float_format="%.12g")
    summary_b.to_csv(args.output_dir / "online_launch_panel_b.csv", index=False, float_format="%.12g")

    print("ONLINE_LAUNCH_PANEL_A_BEGIN")
    print(summary_a.to_csv(index=False), end="")
    print("ONLINE_LAUNCH_PANEL_A_END")
    print("ONLINE_LAUNCH_PANEL_B_BEGIN")
    print(summary_b.to_csv(index=False), end="")
    print("ONLINE_LAUNCH_PANEL_B_END")
    print("NOTE: descriptive pre/post comparison only; no causal interpretation.")


if __name__ == "__main__":
    main()
