"""Calibration and weighting robustness checks promised by the research plan.

This module is deliberately separate from the co-primary TV decision rule.  It
implements equation (9) on the clean Panel A sample, reports both combination-
equal and race-equal weighting, and reports cluster-robust uncertainty.  Because
each race belongs to exactly one race date, race clusters are nested inside date
clusters; the Cameron--Gelbach--Miller two-way race/date covariance therefore
reduces algebraically to the date-cluster covariance.  We report both the
race-cluster and the equivalent race/date two-way standard error explicitly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data_audit import TARGET_MARKETS
from analysis.main_analysis_core import EPSILON, aggregate_point, normalize_inverse_odds, source_group_index
from analysis.main_analysis_panels import load_market
from analysis.main_analysis_runner import common_race_ids, race_metadata, read_frozen_sample


def _wls_fit(y: np.ndarray, x: np.ndarray, field: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return WLS coefficients, design matrix, residuals, and inverse bread."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    field = np.asarray(field)
    weights = np.asarray(weights, dtype=float)
    if not (len(y) == len(x) == len(field) == len(weights)):
        raise ValueError("calibration arrays have inconsistent lengths")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(x)) or np.any(weights <= 0):
        raise ValueError("invalid calibration data")

    levels = np.sort(np.unique(field))
    columns = [np.ones(len(y)), x]
    for level in levels[1:]:
        columns.append((field == level).astype(float))
    design = np.column_stack(columns)
    root_w = np.sqrt(weights)
    xw = design * root_w[:, None]
    yw = y * root_w
    xtwx = xw.T @ xw
    bread = np.linalg.pinv(xtwx, rcond=1e-12)
    beta = bread @ (xw.T @ yw)
    residual = y - design @ beta
    return beta, design, residual, bread


def _cluster_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    bread: np.ndarray,
) -> np.ndarray:
    """One-way cluster sandwich covariance with a conventional finite-sample correction."""
    clusters = np.asarray(clusters)
    weights = np.asarray(weights, dtype=float)
    unique = pd.unique(clusters)
    n = len(residual)
    k = design.shape[1]
    g = len(unique)
    meat = np.zeros((k, k), dtype=float)
    score_rows = design * (weights * residual)[:, None]
    for value in unique:
        score = score_rows[clusters == value].sum(axis=0)
        meat += np.outer(score, score)
    cov = bread @ meat @ bread
    if g > 1 and n > k:
        cov *= (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    return cov


def calibration_regression(
    frame: pd.DataFrame,
    *,
    weighting: str,
) -> dict[str, float | int | str]:
    """Fit equation (9) and return race/date cluster-robust inference.

    ``weighting='combination_equal'`` assigns every outcome one unit of weight.
    ``weighting='race_equal'`` assigns each race total weight one, so every
    outcome in race r receives weight 1/C_r.
    """
    required = {"log_actual", "log_predicted", "n_valid_horses", "race_id", "race_date", "n_outcomes"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"calibration frame missing columns: {sorted(missing)}")
    if weighting == "combination_equal":
        weights = np.ones(len(frame), dtype=float)
    elif weighting == "race_equal":
        weights = 1.0 / frame["n_outcomes"].to_numpy(dtype=float)
    else:
        raise ValueError(f"unknown calibration weighting: {weighting}")

    y = frame["log_actual"].to_numpy(dtype=float)
    x = frame["log_predicted"].to_numpy(dtype=float)
    field = frame["n_valid_horses"].to_numpy()
    beta, design, residual, bread = _wls_fit(y, x, field, weights)
    race = frame["race_id"].astype(str).to_numpy()
    date = frame["race_date"].astype(str).to_numpy()
    mapping = frame[["race_id", "race_date"]].drop_duplicates()
    if mapping["race_id"].duplicated().any():
        raise ValueError("race_id maps to more than one race date")
    cov_race = _cluster_covariance(design, residual, weights, race, bread)
    cov_date = _cluster_covariance(design, residual, weights, date, bread)

    # Race is nested within date, so V_race + V_date - V_race∩date = V_date.
    se_race = float(np.sqrt(max(cov_race[1, 1], 0.0)))
    se_two_way = float(np.sqrt(max(cov_date[1, 1], 0.0)))
    fitted = design @ beta
    weighted_mean = float(np.average(y, weights=weights))
    sse = float(np.sum(weights * (y - fitted) ** 2))
    sst = float(np.sum(weights * (y - weighted_mean) ** 2))
    r2 = float("nan") if sst <= 0 else 1.0 - sse / sst
    return {
        "weighting": weighting,
        "n_races": int(frame["race_id"].nunique()),
        "n_dates": int(frame["race_date"].nunique()),
        "n_combinations": int(len(frame)),
        "alpha_baseline_field_size": float(beta[0]),
        "beta": float(beta[1]),
        "beta_se_race_cluster": se_race,
        "beta_se_race_date_two_way": se_two_way,
        "beta_ci_low_race_cluster": float(beta[1] - 1.96 * se_race),
        "beta_ci_high_race_cluster": float(beta[1] + 1.96 * se_race),
        "beta_ci_low_race_date_two_way": float(beta[1] - 1.96 * se_two_way),
        "beta_ci_high_race_date_two_way": float(beta[1] + 1.96 * se_two_way),
        "calibration_r2": r2,
    }


def _race_calibration(log_actual: np.ndarray, log_predicted: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack([np.ones(len(log_actual)), log_predicted])
    coef, *_ = np.linalg.lstsq(design, log_actual, rcond=None)
    fitted = design @ coef
    centered = log_actual - log_actual.mean()
    sst = float(centered @ centered)
    resid = log_actual - fitted
    r2 = float("nan") if sst <= 0 else 1.0 - float(resid @ resid) / sst
    return float(coef[0]), float(coef[1]), r2


def build_calibration_frame(data_root: Path, sample_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct Panel A main predictions at combination level and race summaries."""
    sample = read_frozen_sample(sample_csv)
    clean_ids = common_race_ids(sample, "eligible_clean_point_sample")
    clean_set = set(clean_ids)
    races = race_metadata(data_root, clean_set)
    meta = races.set_index("race_id")[["race_date", "n_valid_horses"]]
    trifecta = load_market(data_root, "trifecta", clean_set)

    combo_parts: list[pd.DataFrame] = []
    race_rows: list[dict[str, object]] = []
    for target_name in TARGET_MARKETS:
        target = load_market(data_root, target_name, clean_set)
        for race_id in clean_ids:
            source = trifecta.get(race_id)
            actual_frame = target.get(race_id)
            groups = source_group_index(source, actual_frame, target_name)
            cdim = len(actual_frame)
            q_main = normalize_inverse_odds(source["odds"].to_numpy(dtype=float))
            actual = normalize_inverse_odds(actual_frame["odds"].to_numpy(dtype=float))
            predicted = aggregate_point(q_main, groups, cdim)
            log_actual = np.log(actual + EPSILON)
            log_predicted = np.log(predicted + EPSILON)
            alpha_r, beta_r, r2_r = _race_calibration(log_actual, log_predicted)
            info = meta.loc[race_id]
            race_rows.append(
                {
                    "target_market": target_name,
                    "race_id": race_id,
                    "race_date": str(info["race_date"]),
                    "n_valid_horses": int(info["n_valid_horses"]),
                    "n_outcomes": cdim,
                    "calibration_intercept": alpha_r,
                    "calibration_slope": beta_r,
                    "calibration_r2": r2_r,
                }
            )
            combo_parts.append(
                pd.DataFrame(
                    {
                        "target_market": target_name,
                        "race_id": race_id,
                        "race_date": str(info["race_date"]),
                        "n_valid_horses": int(info["n_valid_horses"]),
                        "n_outcomes": cdim,
                        "log_actual": log_actual,
                        "log_predicted": log_predicted,
                    }
                )
            )
    return pd.concat(combo_parts, ignore_index=True), pd.DataFrame(race_rows)


def summarize_calibration(combo: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGET_MARKETS:
        c = combo[combo["target_market"].eq(target)].copy()
        r = race[race["target_market"].eq(target)].copy()
        for weighting in ("race_equal", "combination_equal"):
            record = calibration_regression(c, weighting=weighting)
            record.update(
                {
                    "target_market": target,
                    "median_race_calibration_intercept": float(r["calibration_intercept"].median()),
                    "median_race_calibration_slope": float(r["calibration_slope"].median()),
                    "median_race_calibration_r2": float(r["calibration_r2"].median()),
                    "share_race_slope_0_9_1_1": float(r["calibration_slope"].between(0.9, 1.1, inclusive="both").mean()),
                }
            )
            rows.append(record)
    columns = [
        "target_market", "weighting", "n_races", "n_dates", "n_combinations",
        "alpha_baseline_field_size", "beta", "beta_se_race_cluster",
        "beta_se_race_date_two_way", "beta_ci_low_race_cluster",
        "beta_ci_high_race_cluster", "beta_ci_low_race_date_two_way",
        "beta_ci_high_race_date_two_way", "calibration_r2",
        "median_race_calibration_intercept", "median_race_calibration_slope",
        "median_race_calibration_r2", "share_race_slope_0_9_1_1",
    ]
    return pd.DataFrame(rows)[columns]


def write_table(summary: pd.DataFrame, output: Path) -> None:
    labels = {"win": "단승", "exacta": "쌍승", "quinella": "복승", "trio": "삼복승"}
    weight_labels = {"race_equal": "경주균등", "combination_equal": "조합균등"}
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"승식 & 가중 & $\hat\beta$ & 경주군집 SE & 경주--날짜 SE & 경주별 $\beta$ 중앙값 & $[0.9,1.1]$ 비율 \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{labels[row.target_market]} & {weight_labels[row.weighting]} & "
            f"{row.beta:.4f} & {row.beta_se_race_cluster:.4f} & "
            f"{row.beta_se_race_date_two_way:.4f} & "
            f"{row.median_race_calibration_slope:.4f} & "
            f"{100*row.share_race_slope_0_9_1_1:.1f}\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = Path("outputs")
    table_dir = Path("tables")
    combo, race = build_calibration_frame(Path("KRA/parsed"), output_dir / "analysis_sample.csv")
    summary = summarize_calibration(combo, race)
    summary.to_csv(output_dir / "main_calibration_robustness.csv", index=False, float_format="%.12g")
    race.to_csv(output_dir / "main_calibration_by_race.csv", index=False, float_format="%.12g")
    write_table(summary, table_dir / "main_calibration_robustness.tex")
    print(summary.to_string(index=False))
    print("PASS: calibration and weighting robustness generated")


if __name__ == "__main__":
    main()
