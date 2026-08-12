"""Command-line runner for the co-primary cross-pool main analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data_audit import TARGET_MARKETS, parse_horse_list, prepare_races
from analysis.main_analysis_core import SOURCE_MARKET
from analysis.main_analysis_donor_reuse import donor_reuse_diagnostic
from analysis.main_analysis_guards import assert_win_uncapped
from analysis.main_analysis_p3 import (
    order_information_bounds,
    write_order_information_bounds_table,
)
from analysis.main_analysis_panels import (
    grouped_ids_by_field,
    load_market,
    panel_a,
    panel_b,
)
from analysis.main_analysis_report import (
    absolute_threshold_decision,
    benchmark_improvements_a,
    benchmark_improvements_b,
    external_log_score_summary,
    order_information_test,
    sample_composition,
    sample_selection_summary,
    summarize_panel_a,
    summarize_panel_b,
    write_latex_tables,
    write_manifest,
    write_panel_b_subset_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("KRA/parsed"))
    parser.add_argument("--sample-csv", type=Path, default=Path("outputs/analysis_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))
    parser.add_argument("--max-races", type=int, default=0, help="deterministic development limit")
    parser.add_argument("--skip-bounds", action="store_true", help="Panel A only; development/smoke use")
    return parser.parse_args()


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    mapped = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if mapped.isna().any():
        raise ValueError("sample membership column contains non-boolean values")
    return mapped.astype(bool)


def read_frozen_sample(sample_csv: Path) -> pd.DataFrame:
    if not sample_csv.exists():
        raise FileNotFoundError(
            f"{sample_csv} is missing; run `python -m analysis.data_audit --strict` first"
        )
    sample = pd.read_csv(sample_csv, low_memory=False)
    required = {
        "race_id",
        "target_market",
        "eligible_complete_sample",
        "eligible_clean_point_sample",
        "eligible_capped_interval_sample",
    }
    missing = required.difference(sample.columns)
    if missing:
        raise ValueError(f"analysis sample missing columns: {sorted(missing)}")
    return sample


def target_race_ids(sample: pd.DataFrame, target: str, column: str) -> list[str]:
    mask = _boolean_series(sample[column])
    rows = sample[sample["target_market"].eq(target) & mask]
    return sorted(rows["race_id"].astype(str).tolist())


def common_race_ids(sample: pd.DataFrame, column: str) -> list[str]:
    sets = [set(target_race_ids(sample, target, column)) for target in TARGET_MARKETS]
    if not all(values == sets[0] for values in sets[1:]):
        raise ValueError(f"target samples differ for {column}")
    return sorted(sets[0])


def race_metadata(data_root: Path, race_ids: set[str]) -> pd.DataFrame:
    races = prepare_races(data_root)
    races = races[races["race_id"].isin(race_ids)].copy()
    races["year"] = pd.to_datetime(races["race_date"]).dt.year
    races["valid_horse_tuple"] = races["valid_horses"].map(parse_horse_list)
    races["arrival_tuple"] = races["arrival_order"].map(parse_horse_list)
    invalid_arrivals = races["arrival_tuple"].map(
        lambda value: len(value) < 3 or len(set(value[:3])) != 3
    )
    if bool(invalid_arrivals.any()):
        raise ValueError("race metadata contains an invalid top-three arrival order")
    return races


def tail_summary(
    trifecta_frame: pd.DataFrame, clean_ids: set[str], full_ids: set[str]
) -> pd.DataFrame:
    frame = trifecta_frame[trifecta_frame["race_id"].isin(full_ids)].copy()
    frame["sample_group"] = np.where(frame["race_id"].isin(clean_ids), "clean", "capped")
    frame = frame[~frame["is_capped_odds"].fillna(False)].copy()
    rows: list[dict[str, object]] = []
    for group, values in frame.groupby("sample_group")["odds"]:
        numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
        rows.append(
            {
                "sample_group": group,
                "n_uncapped_combinations": len(numeric),
                "odds_q90": float(numeric.quantile(0.90)),
                "odds_q95": float(numeric.quantile(0.95)),
                "odds_q99": float(numeric.quantile(0.99)),
                "odds_max": float(numeric.max()),
            }
        )
    return pd.DataFrame(rows)


def heterogeneity_summary(metrics: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    metadata = races[["race_id", "year", "meet", "n_valid_horses"]].copy()
    merged = metrics.merge(metadata, on=["race_id", "n_valid_horses"], how="left")
    frames: list[pd.DataFrame] = []
    for dimension in ("year", "meet", "n_valid_horses"):
        grouped = (
            merged.groupby(["target_market", "model", dimension], dropna=False)["tv"]
            .agg(
                n_races="size",
                median_tv="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            )
            .reset_index()
        )
        grouped["dimension"] = dimension
        grouped["level"] = grouped[dimension].astype(str)
        frames.append(
            grouped[
                [
                    "target_market",
                    "model",
                    "dimension",
                    "level",
                    "n_races",
                    "median_tv",
                    "q25",
                    "q75",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    sample = read_frozen_sample(args.sample_csv)
    all_clean_ids = common_race_ids(sample, "eligible_clean_point_sample")
    all_full_ids = common_race_ids(sample, "eligible_complete_sample")
    clean_ids = all_clean_ids[: args.max_races] if args.max_races > 0 else all_clean_ids
    full_ids = all_full_ids[: args.max_races] if args.max_races > 0 else all_full_ids
    full_ids = sorted(set(full_ids).union(clean_ids))
    if not clean_ids or not full_ids:
        raise ValueError("analysis sample is empty")

    # Donor benchmarks are defined using the frozen panel, not an arbitrary
    # development subset. Loading all source races keeps --max-races from
    # changing donor eligibility or assignment.
    source_ids = set(all_full_ids)
    races = race_metadata(args.data_root, source_ids)
    if len(races) != len(source_ids):
        raise ValueError("race metadata does not cover the frozen analysis sample")
    trifecta = load_market(args.data_root, SOURCE_MARKET, source_ids)
    win = load_market(args.data_root, "win", source_ids)
    assert_win_uncapped(win.frame)
    clean_peers = grouped_ids_by_field(races, all_clean_ids)
    full_peers = grouped_ids_by_field(races, all_full_ids)

    panel_a_frames: list[pd.DataFrame] = []
    panel_b_frames: list[pd.DataFrame] = []
    for target_name in TARGET_MARKETS:
        target = load_market(args.data_root, target_name, source_ids)
        panel_a_frames.append(
            panel_a(
                races,
                trifecta,
                win,
                target_name,
                target,
                clean_ids,
                clean_peers,
            )
        )
        if not args.skip_bounds:
            panel_b_frames.append(
                panel_b(
                    races,
                    trifecta,
                    win,
                    target_name,
                    target,
                    full_ids,
                    full_peers,
                )
            )

    metrics_a = pd.concat(panel_a_frames, ignore_index=True)
    bounds_b = None if args.skip_bounds else pd.concat(panel_b_frames, ignore_index=True)
    summary_a = summarize_panel_a(metrics_a)
    log_scores = external_log_score_summary(metrics_a)
    summary_b = None if bounds_b is None else summarize_panel_b(bounds_b)
    capped_bounds_b = (
        None
        if bounds_b is None
        else bounds_b[bounds_b["race_id"].isin(set(full_ids) - set(clean_ids))].copy()
    )
    capped_summary_b = (
        None if capped_bounds_b is None else summarize_panel_b(capped_bounds_b)
    )
    improve_a = benchmark_improvements_a(metrics_a)
    improve_b = None if bounds_b is None else benchmark_improvements_b(bounds_b)
    capped_improve_b = (
        None
        if capped_bounds_b is None
        else benchmark_improvements_b(capped_bounds_b)
    )
    p3 = order_information_test(metrics_a)
    p3_bounds = None if bounds_b is None else order_information_bounds(bounds_b)
    thresholds = absolute_threshold_decision(summary_a, summary_b)

    cap_by_race = trifecta.frame.groupby("race_id")["is_capped_odds"].sum().astype(int)
    selection = sample_selection_summary(
        races, set(all_clean_ids), set(all_full_ids), cap_by_race
    )
    composition = sample_composition(races, set(all_clean_ids), set(all_full_ids))
    tails = tail_summary(trifecta.frame, set(all_clean_ids), set(all_full_ids))
    heterogeneity = heterogeneity_summary(metrics_a, races)
    donor_reuse = donor_reuse_diagnostic(metrics_a, bounds_b)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    outputs: list[tuple[Path, pd.DataFrame]] = [
        (args.output_dir / "main_metrics.csv", metrics_a),
        (args.output_dir / "main_panel_a_summary.csv", summary_a),
        (args.output_dir / "main_external_log_scores.csv", log_scores),
        (args.output_dir / "main_panel_a_improvements.csv", improve_a),
        (args.output_dir / "main_order_information.csv", p3),
        (args.output_dir / "main_threshold_decisions.csv", thresholds),
        (args.output_dir / "main_sample_selection.csv", selection),
        (args.output_dir / "main_sample_composition.csv", composition),
        (args.output_dir / "main_sample_tail.csv", tails),
        (args.output_dir / "main_heterogeneity.csv", heterogeneity),
        (args.output_dir / "main_other_race_donor_reuse.csv", donor_reuse),
    ]
    if bounds_b is not None and summary_b is not None and improve_b is not None:
        outputs.extend(
            [
                (args.output_dir / "main_metrics_bounds.csv", bounds_b),
                (args.output_dir / "main_panel_b_summary.csv", summary_b),
                (args.output_dir / "main_panel_b_improvements.csv", improve_b),
            ]
        )
    if capped_summary_b is not None and capped_improve_b is not None:
        outputs.extend(
            [
                (args.output_dir / "main_panel_b_capped_summary.csv", capped_summary_b),
                (
                    args.output_dir / "main_panel_b_capped_improvements.csv",
                    capped_improve_b,
                ),
            ]
        )
    if p3_bounds is not None:
        outputs.append((args.output_dir / "main_order_information_bounds.csv", p3_bounds))
    for path, frame in outputs:
        frame.to_csv(path, index=False, float_format="%.12g")
        written_rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
        if written_rows != len(frame):
            raise IOError(
                f"incomplete CSV write for {path}: expected {len(frame)} rows, "
                f"found {written_rows}"
            )
        generated.append(path)

    generated.extend(
        write_latex_tables(
            args.table_dir,
            summary_a,
            summary_b,
            improve_a,
            improve_b,
            p3,
            log_scores,
        )
    )
    if p3_bounds is not None:
        generated.append(write_order_information_bounds_table(args.table_dir, p3_bounds))
    if capped_summary_b is not None:
        generated.append(write_panel_b_subset_table(args.table_dir, capped_summary_b))
    write_manifest(
        args.output_dir,
        generated,
        max_races=args.max_races,
        bounds=not args.skip_bounds,
    )

    print(f"Panel A evaluated races: {len(clean_ids):,} / frozen {len(all_clean_ids):,}")
    print(
        f"Panel B evaluated races: {0 if args.skip_bounds else len(full_ids):,} / "
        f"frozen {len(all_full_ids):,}"
    )
    print(summary_a[summary_a["model"].eq("main")].to_string(index=False))
    if summary_b is not None:
        print(summary_b[summary_b["model"].eq("main")].to_string(index=False))
    if p3_bounds is not None:
        print(p3_bounds.to_string(index=False))
    print(donor_reuse.to_string(index=False))
    print("PASS: main cross-pool analysis completed")
