"""Summaries, pre-registered decisions, and manuscript tables for the main analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data_audit import TARGET_MARKETS
from analysis.main_analysis_core import (
    BOOTSTRAP_REPS,
    DISPLAY_CAP,
    EPSILON,
    MODELS,
    RANDOM_SEED,
    ROUNDING_HALF_WIDTH,
    TV_SENSITIVITY_THRESHOLDS,
    TV_THRESHOLD,
    stable_uint,
)


def bootstrap_ci(
    values: np.ndarray, *, reps: int = BOOTSTRAP_REPS, label: str
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.median(values))
    if len(values) == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(stable_uint(f"bootstrap|{label}"))
    draws = np.empty(reps, dtype=float)
    n = len(values)
    for b in range(reps):
        draws[b] = np.median(values[rng.integers(0, n, size=n)])
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def summarize_panel_a(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, model), group in metrics.groupby(["target_market", "model"], sort=True):
        tv_med, tv_lo, tv_hi = bootstrap_ci(
            group["tv"].to_numpy(), label=f"A|{target}|{model}|tv"
        )
        js_med, js_lo, js_hi = bootstrap_ci(
            group["js"].to_numpy(), label=f"A|{target}|{model}|js"
        )
        rows.append(
            {
                "panel": "A",
                "target_market": target,
                "model": model,
                "n_races": len(group),
                "median_tv": tv_med,
                "median_tv_ci_low": tv_lo,
                "median_tv_ci_high": tv_hi,
                "median_js": js_med,
                "median_js_ci_low": js_lo,
                "median_js_ci_high": js_hi,
                "median_mae": float(group["mae"].median()),
                "median_r2": float(group["r2"].median()),
                "share_r2_gt_0_8": float((group["r2"] > 0.8).mean()),
                "median_rmsle": float(group["rmsle"].median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_panel_b(bounds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, model), group in bounds.groupby(["target_market", "model"], sort=True):
        low_med, low_lo, low_hi = bootstrap_ci(
            group["tv_lower"].to_numpy(), label=f"B|{target}|{model}|low"
        )
        up_med, up_lo, up_hi = bootstrap_ci(
            group["tv_upper_outer"].to_numpy(), label=f"B|{target}|{model}|up"
        )
        rows.append(
            {
                "panel": "B",
                "target_market": target,
                "model": model,
                "n_races": len(group),
                "median_tv_lower": low_med,
                "median_tv_lower_ci_low": low_lo,
                "median_tv_lower_ci_high": low_hi,
                "median_tv_upper_outer": up_med,
                "median_tv_upper_outer_ci_low": up_lo,
                "median_tv_upper_outer_ci_high": up_hi,
            }
        )
    return pd.DataFrame(rows)


def _paired_improvement(
    pivot_a: pd.DataFrame,
    pivot_b: pd.DataFrame,
    target: str,
    benchmark: str,
) -> np.ndarray:
    left = pivot_a.xs(target, level="target_market")[[benchmark]].rename(
        columns={benchmark: "benchmark"}
    )
    right = pivot_b.xs(target, level="target_market")[["main"]]
    joined = left.join(right, how="inner").dropna()
    return (joined["benchmark"] - joined["main"]).to_numpy(dtype=float)


def benchmark_improvements_a(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(index=["race_id", "target_market"], columns="model", values="tv")
    rows: list[dict[str, object]] = []
    for target in TARGET_MARKETS:
        for benchmark in MODELS[1:]:
            if benchmark not in pivot.columns:
                continue
            delta = _paired_improvement(pivot, pivot, target, benchmark)
            med, lo, hi = bootstrap_ci(delta, label=f"A|improvement|{target}|{benchmark}")
            rows.append(
                {
                    "panel": "A",
                    "target_market": target,
                    "benchmark": benchmark,
                    "n_races": len(delta),
                    "median_improvement_lower": med,
                    "ci_low": lo,
                    "ci_high": hi,
                    "main_better": bool(np.isfinite(lo) and lo > 0),
                }
            )
    return pd.DataFrame(rows)


def benchmark_improvements_b(bounds: pd.DataFrame) -> pd.DataFrame:
    lower = bounds.pivot(
        index=["race_id", "target_market"], columns="model", values="tv_lower"
    )
    upper = bounds.pivot(
        index=["race_id", "target_market"], columns="model", values="tv_upper_outer"
    )
    rows: list[dict[str, object]] = []
    for target in TARGET_MARKETS:
        for benchmark in MODELS[1:]:
            if benchmark not in lower.columns:
                continue
            delta = _paired_improvement(lower, upper, target, benchmark)
            med, lo, hi = bootstrap_ci(delta, label=f"B|improvement|{target}|{benchmark}")
            rows.append(
                {
                    "panel": "B",
                    "target_market": target,
                    "benchmark": benchmark,
                    "n_races": len(delta),
                    "median_improvement_lower": med,
                    "ci_low": lo,
                    "ci_high": hi,
                    "main_better": bool(np.isfinite(lo) and lo > 0),
                    "share_races_robust_main_better": float(np.mean(delta > 0)) if len(delta) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def order_information_test(metrics: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for measure in ("tv", "js"):
        main = metrics[metrics["model"].eq("main")].pivot(
            index="race_id", columns="target_market", values=measure
        )
        harville = metrics[metrics["model"].eq("harville")].pivot(
            index="race_id", columns="target_market", values=measure
        )
        common = main.index.intersection(harville.index)
        exacta_gain = harville.loc[common, "exacta"] - main.loc[common, "exacta"]
        quinella_gain = harville.loc[common, "quinella"] - main.loc[common, "quinella"]
        difference = (exacta_gain - quinella_gain).to_numpy(dtype=float)
        med, lo, hi = bootstrap_ci(difference, label=f"P3|{measure}")
        out.append(
            {
                "measure": measure,
                "n_races": len(difference),
                "median_exacta_gain": float(np.median(exacta_gain)),
                "median_quinella_gain": float(np.median(quinella_gain)),
                "median_difference": med,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(out)


def absolute_threshold_decision(
    summary_a: pd.DataFrame, summary_b: pd.DataFrame | None
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    panel_a = summary_a[summary_a["model"].eq("main")].set_index("target_market")
    panel_b = (
        None
        if summary_b is None
        else summary_b[summary_b["model"].eq("main")].set_index("target_market")
    )
    for target in TARGET_MARKETS:
        for threshold in (TV_THRESHOLD, *TV_SENSITIVITY_THRESHOLDS):
            a_pass = bool(panel_a.loc[target, "median_tv_ci_high"] < threshold)
            b_pass = (
                False
                if panel_b is None
                else bool(panel_b.loc[target, "median_tv_upper_outer_ci_high"] < threshold)
            )
            rows.append(
                {
                    "target_market": target,
                    "threshold": threshold,
                    "panel_a_pass": a_pass,
                    "panel_b_pass": b_pass,
                    "co_primary_pass": bool(a_pass and b_pass),
                }
            )
    return pd.DataFrame(rows)


def sample_selection_summary(
    races: pd.DataFrame,
    clean_ids: set[str],
    full_ids: set[str],
    trifecta_capped_rows: pd.Series | None = None,
) -> pd.DataFrame:
    subset = races[races["race_id"].isin(full_ids)].copy()
    subset["sample_group"] = np.where(subset["race_id"].isin(clean_ids), "clean", "capped")
    if trifecta_capped_rows is not None:
        subset = subset.merge(
            trifecta_capped_rows.rename("trifecta_capped_rows"),
            left_on="race_id",
            right_index=True,
            how="left",
        )
    rows: list[dict[str, object]] = []
    for group, frame in subset.groupby("sample_group"):
        field = frame["n_valid_horses"].astype(float)
        rows.append(
            {
                "sample_group": group,
                "n_races": len(frame),
                "field_size_median": float(field.median()),
                "field_size_q25": float(field.quantile(0.25)),
                "field_size_q75": float(field.quantile(0.75)),
                "first_year": int(frame["year"].min()),
                "last_year": int(frame["year"].max()),
                "n_meets": int(frame["meet"].nunique()),
                "trifecta_capped_rows_median": (
                    float(frame["trifecta_capped_rows"].median())
                    if "trifecta_capped_rows" in frame
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def sample_composition(races: pd.DataFrame, clean_ids: set[str], full_ids: set[str]) -> pd.DataFrame:
    """Expose year/track/field-size composition so clean-sample selection is visible."""
    subset = races[races["race_id"].isin(full_ids)].copy()
    subset["sample_group"] = np.where(subset["race_id"].isin(clean_ids), "clean", "capped")
    frames: list[pd.DataFrame] = []
    for variable in ("year", "meet", "n_valid_horses"):
        counts = (
            subset.groupby(["sample_group", variable], dropna=False)
            .size()
            .rename("n_races")
            .reset_index()
        )
        counts["dimension"] = variable
        counts["level"] = counts[variable].astype(str)
        counts = counts[["sample_group", "dimension", "level", "n_races"]]
        totals = counts.groupby("sample_group")["n_races"].transform("sum")
        counts["share"] = counts["n_races"] / totals
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def fmt(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return latex_escape(value)


def write_latex_tables(
    table_dir: Path,
    summary_a: pd.DataFrame,
    summary_b: pd.DataFrame | None,
    improvement_a: pd.DataFrame,
    improvement_b: pd.DataFrame | None,
    p3: pd.DataFrame,
) -> list[Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"승식 & 모형 & 경주 수 & 중앙 TV & 95\% CI 하한 & 95\% CI 상한 \\",
        r"\midrule",
    ]
    for _, row in summary_a.iterrows():
        lines.append(
            f"{fmt(row['target_market'])} & {fmt(row['model'])} & {fmt(row['n_races'])} & "
            f"{fmt(row['median_tv'])} & {fmt(row['median_tv_ci_low'])} & {fmt(row['median_tv_ci_high'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = table_dir / "main_panel_a.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(path)

    if summary_b is not None:
        lines = [
            r"\begin{tabular}{llrrrrr}",
            r"\toprule",
            r"승식 & 모형 & 경주 수 & TV 하한 중앙값 & 하한 95\% CI & TV 상한 중앙값 & 상한 95\% CI \\",
            r"\midrule",
        ]
        for _, row in summary_b.iterrows():
            lines.append(
                f"{fmt(row['target_market'])} & {fmt(row['model'])} & {fmt(row['n_races'])} & "
                f"{fmt(row['median_tv_lower'])} & [{fmt(row['median_tv_lower_ci_low'])}, {fmt(row['median_tv_lower_ci_high'])}] & "
                f"{fmt(row['median_tv_upper_outer'])} & [{fmt(row['median_tv_upper_outer_ci_low'])}, {fmt(row['median_tv_upper_outer_ci_high'])}] \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        path = table_dir / "main_panel_b.tex"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)

    merged = improvement_a.copy()
    if improvement_b is not None:
        merged = pd.concat([merged, improvement_b], ignore_index=True)
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"패널 & 승식 & 기준모형 & 경주 수 & 개선폭 중앙값 & 95\% CI & 우위 판정 \\",
        r"\midrule",
    ]
    for _, row in merged.iterrows():
        lines.append(
            f"{fmt(row['panel'])} & {fmt(row['target_market'])} & {fmt(row['benchmark'])} & {fmt(row['n_races'])} & "
            f"{fmt(row['median_improvement_lower'])} & [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] & {fmt(row['main_better'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = table_dir / "main_benchmark_comparison.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(path)

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"거리 & 경주 수 & 쌍승 개선 & 복승 개선 & 차이 중앙값 & 95\% CI 하한 & 95\% CI 상한 \\",
        r"\midrule",
    ]
    for _, row in p3.iterrows():
        lines.append(
            f"{fmt(row['measure'])} & {fmt(row['n_races'])} & {fmt(row['median_exacta_gain'])} & "
            f"{fmt(row['median_quinella_gain'])} & {fmt(row['median_difference'])} & {fmt(row['ci_low'])} & {fmt(row['ci_high'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = table_dir / "main_order_information.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(path)
    return written


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    output_dir: Path,
    generated: list[Path],
    *,
    max_races: int,
    bounds: bool,
) -> Path:
    payload = {
        "schema_version": 2,
        "analysis": "cross_pool_main_analysis",
        "random_seed": RANDOM_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "rounding_half_width": ROUNDING_HALF_WIDTH,
        "display_cap": DISPLAY_CAP,
        "epsilon": EPSILON,
        "tv_threshold": TV_THRESHOLD,
        "tv_sensitivity_thresholds": list(TV_SENSITIVITY_THRESHOLDS),
        "panel_b_bounds_computed": bounds,
        "max_races": max_races,
        "files": {},
    }
    for path in generated:
        payload["files"][str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    manifest = output_dir / "main_analysis_manifest.json"
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
