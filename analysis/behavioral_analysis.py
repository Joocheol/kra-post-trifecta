#!/usr/bin/env python3
"""Run the year-cross-fitted rank-probability and behavioral price analysis.

The default run uses all in-scope races.  Exacta and trifecta comparisons use
market-specific complete, uncapped price vectors: this retains 18,703 exacta
races and 3,321 trifecta races in the frozen data.  Capped odds are never treated
as point observations in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analysis.behavioral_core import (
    EPSILON,
    MonotoneCalibrator,
    PowerWeight,
    PrelecWeight,
    StageTemperature,
    expected_calibration_error,
    fit_stage_temperature,
)
from analysis.data_audit import (
    EXCLUDED_DATES,
    EXCLUDED_YEARS,
    MARKET_SPECS,
    START_DATE,
    END_DATE,
    parquet_paths,
    prepare_races,
    read_parquets,
)


PROBABILITY_MODELS = ("harville", "stage_temperature")
TAIL_MODELS = ("isotonic_clip", "prelec", "power")
PRICE_MODELS = {
    "exacta": ("M-U", "M-R", "M-S2"),
    "trifecta": ("M-U", "M-R", "M-S2", "M-S3"),
}
TARGET_MARKETS = tuple(PRICE_MODELS)
COMMON_SUPPORT_QUANTILES = (0.01, 0.99)
BOOTSTRAP_REPS = 999


@dataclass(frozen=True)
class FoldModel:
    validation_year: int
    calibration: MonotoneCalibrator
    stage2: StageTemperature
    stage3: StageTemperature
    win_weight: MonotoneCalibrator
    prelec: PrelecWeight
    power: PowerWeight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("KRA/parsed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--max-races",
        type=int,
        default=0,
        help="development-only deterministic cap, allocated across years",
    )
    parser.add_argument(
        "--skip-price-models",
        action="store_true",
        help="run only the rank-probability cross-validation stage",
    )
    return parser.parse_args()


def parse_arrival(value: object) -> tuple[int, ...]:
    if value is None or pd.isna(value):
        return ()
    try:
        return tuple(int(part) for part in str(value).split(",") if part)
    except ValueError:
        return ()


def deterministic_race_cap(races: pd.DataFrame, max_races: int) -> pd.DataFrame:
    if max_races <= 0 or max_races >= len(races):
        return races.copy()
    years = sorted(races["year"].unique())
    base, remainder = divmod(max_races, len(years))
    pieces: list[pd.DataFrame] = []
    for index, year in enumerate(years):
        count = base + (index < remainder)
        pieces.append(
            races[races["year"].eq(year)].sort_values("race_id").head(count)
        )
    result = pd.concat(pieces, ignore_index=True).sort_values("race_id")
    if len(result) != max_races:
        raise ValueError("development race cap could not be allocated across years")
    return result.reset_index(drop=True)


def load_horse_panel(data_root: Path, max_races: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    races = prepare_races(data_root)
    races = races[races["in_date_scope"]].copy()
    races["year"] = races["race_date"].str[:4].astype(int)
    races["arrival"] = races["arrival_order"].map(parse_arrival)
    races["top1"] = races["arrival"].map(lambda value: value[0] if len(value) >= 1 else -1)
    races["top2"] = races["arrival"].map(lambda value: value[1] if len(value) >= 2 else -1)
    races["top3"] = races["arrival"].map(lambda value: value[2] if len(value) >= 3 else -1)
    valid_top3 = races.apply(
        lambda row: len(set(row["arrival"][:3])) == 3
        and set(row["arrival"][:3]).issubset(set(row["valid_horse_tuple"])),
        axis=1,
    )
    if not bool(valid_top3.all()):
        raise ValueError(f"{int((~valid_top3).sum())} races have invalid top-three finishers")
    races = deterministic_race_cap(races, max_races)

    win = read_parquets(
        data_root,
        "win",
        columns=[
            "race_id",
            "horse_no",
            "odds",
            "is_hit",
            "is_capped_odds",
            "race_date",
            "meet",
        ],
    )
    race_columns = [
        "race_id",
        "year",
        "n_valid_horses",
        "top1",
        "top2",
        "top3",
    ]
    horses = win.merge(races[race_columns], on="race_id", how="inner", validate="many_to_one")
    if len(horses) != int(races["n_valid_horses"].sum()):
        raise ValueError("win rows do not match the valid-horse counts")
    if bool(horses["is_capped_odds"].any()):
        raise ValueError("win odds unexpectedly contain capped observations")
    if bool(horses["odds"].isna().any()) or bool((horses["odds"] <= 0).any()):
        raise ValueError("win odds contain missing or non-positive values")
    hit_count = horses.groupby("race_id")["is_hit"].sum()
    if not bool(hit_count.eq(1).all()):
        raise ValueError("each race must contain exactly one winning horse")
    winner_matches = horses.loc[horses["is_hit"], "horse_no"].to_numpy() == races.set_index(
        "race_id"
    ).loc[horses.loc[horses["is_hit"], "race_id"], "top1"].to_numpy()
    if not bool(winner_matches.all()):
        raise ValueError("win hit flags disagree with arrival order")

    horses["inverse_odds"] = 1.0 / horses["odds"].astype(float)
    horses["win_price_share"] = horses["inverse_odds"] / horses.groupby("race_id")[
        "inverse_odds"
    ].transform("sum")
    horses["race_equal_weight"] = 1.0 / horses["n_valid_horses"]
    return races.reset_index(drop=True), horses.sort_values(
        ["race_id", "horse_no"]
    ).reset_index(drop=True)


def apply_calibration(horses: pd.DataFrame, fit: MonotoneCalibrator) -> pd.DataFrame:
    result = horses.copy()
    result["objective_score"] = fit.predict(result["win_price_share"].to_numpy())
    result["objective_probability"] = result["objective_score"] / result.groupby(
        "race_id"
    )["objective_score"].transform("sum")
    return result


def stage_choice_sets(
    horses: pd.DataFrame,
    stage: int,
) -> list[tuple[np.ndarray, int]]:
    if stage not in (2, 3):
        raise ValueError("only stages two and three have fitted temperatures")
    result: list[tuple[np.ndarray, int]] = []
    for _, group in horses.groupby("race_id", sort=False):
        top1 = int(group["top1"].iloc[0])
        top2 = int(group["top2"].iloc[0])
        chosen = int(group[f"top{stage}"].iloc[0])
        excluded = {top1} if stage == 2 else {top1, top2}
        remaining = group.loc[~group["horse_no"].isin(excluded)].reset_index(drop=True)
        matches = np.flatnonzero(remaining["horse_no"].to_numpy() == chosen)
        if len(matches) != 1:
            raise ValueError("realized finisher is absent or duplicated in a choice set")
        result.append((remaining["objective_probability"].to_numpy(), int(matches[0])))
    return result


def fit_fold(train: pd.DataFrame, validation_year: int) -> tuple[FoldModel, pd.DataFrame]:
    calibration = MonotoneCalibrator.fit(
        train["win_price_share"].to_numpy(),
        train["is_hit"].astype(float).to_numpy(),
        train["race_equal_weight"].to_numpy(),
    )
    calibrated_train = apply_calibration(train, calibration)
    stage2 = fit_stage_temperature(stage_choice_sets(calibrated_train, 2))
    stage3 = fit_stage_temperature(stage_choice_sets(calibrated_train, 3))

    weight = calibrated_train["race_equal_weight"].to_numpy()
    probability = calibrated_train["objective_probability"].to_numpy()
    price = calibrated_train["win_price_share"].to_numpy()
    support_lower, support_upper = np.quantile(probability, COMMON_SUPPORT_QUANTILES)
    supported = (probability >= support_lower) & (probability <= support_upper)
    win_weight = MonotoneCalibrator.fit(
        probability[supported], price[supported], weight[supported]
    )
    prelec = PrelecWeight.fit(probability, price, weight)
    power = PowerWeight.fit(probability, price, weight)
    return (
        FoldModel(
            validation_year=validation_year,
            calibration=calibration,
            stage2=stage2,
            stage3=stage3,
            win_weight=win_weight,
            prelec=prelec,
            power=power,
        ),
        calibrated_train,
    )


def model_temperatures(fold: FoldModel, probability_model: str) -> tuple[StageTemperature, StageTemperature]:
    if probability_model == "harville":
        identity = StageTemperature(0.0, 0.0)
        return identity, identity
    if probability_model == "stage_temperature":
        return fold.stage2, fold.stage3
    raise ValueError(f"unknown probability model: {probability_model}")


def rank_validation(
    validation: pd.DataFrame,
    fold: FoldModel,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in PROBABILITY_MODELS:
        stage2, stage3 = model_temperatures(fold, model_name)
        for stage, temperature in ((1, None), (2, stage2), (3, stage3)):
            log_losses: list[float] = []
            brier_scores: list[float] = []
            all_probability: list[np.ndarray] = []
            all_outcome: list[np.ndarray] = []
            for _, group in validation.groupby("race_id", sort=False):
                excluded: set[int] = set()
                if stage >= 2:
                    excluded.add(int(group["top1"].iloc[0]))
                if stage >= 3:
                    excluded.add(int(group["top2"].iloc[0]))
                chosen = int(group[f"top{stage}"].iloc[0])
                remaining = group.loc[~group["horse_no"].isin(excluded)]
                strength = remaining["objective_probability"].to_numpy()
                if temperature is not None:
                    remaining_size = int(group["n_valid_horses"].iloc[0]) - stage + 1
                    alpha = float(temperature.alpha(remaining_size))
                    strength = np.power(strength, alpha)
                probability = strength / strength.sum()
                outcome = remaining["horse_no"].eq(chosen).astype(float).to_numpy()
                if int(outcome.sum()) != 1:
                    raise ValueError("validation choice set has no unique realized outcome")
                chosen_probability = float(probability[outcome.astype(bool)][0])
                log_losses.append(-np.log(max(chosen_probability, EPSILON)))
                brier_scores.append(float(np.square(probability - outcome).sum()))
                all_probability.append(probability)
                all_outcome.append(outcome)
            rows.append(
                {
                    "validation_year": fold.validation_year,
                    "probability_model": model_name,
                    "stage": stage,
                    "n_races": int(validation["race_id"].nunique()),
                    "mean_log_loss": float(np.mean(log_losses)),
                    "mean_brier": float(np.mean(brier_scores)),
                    "ece_10": expected_calibration_error(
                        np.concatenate(all_probability), np.concatenate(all_outcome), bins=10
                    ),
                }
            )
    return pd.DataFrame(rows)


def read_market_year(
    data_root: Path,
    market: str,
    year: int,
    columns: list[str],
) -> pd.DataFrame:
    paths = [
        path
        for path in parquet_paths(data_root, market)
        if f"/year={year}/" in path.as_posix()
    ]
    if not paths:
        raise FileNotFoundError(f"no {market} parquets found for {year}")
    return pq.read_table(paths, columns=columns).to_pandas()


def uncapped_target(
    data_root: Path,
    market: str,
    year: int,
    race_ids: set[str],
) -> pd.DataFrame:
    spec = MARKET_SPECS[market]
    columns = ["race_id", *spec.keys, "odds", "is_capped_odds", "race_date", "meet"]
    frame = read_market_year(data_root, market, year, columns)
    frame = frame[frame["race_id"].isin(race_ids)].copy()
    if frame.empty:
        return frame
    capped_race = frame.groupby("race_id")["is_capped_odds"].transform("any")
    frame = frame.loc[~capped_race].copy()
    if frame.empty:
        return frame
    if bool(frame["odds"].isna().any()) or bool((frame["odds"] <= 0).any()):
        raise ValueError(f"{market} contains missing or non-positive odds")
    n_by_race = frame.groupby("race_id").size()
    if market == "exacta":
        implied_n = ((1 + np.sqrt(1 + 4 * n_by_race)) / 2).round().astype(int)
        expected_rows = implied_n * (implied_n - 1)
    elif market == "trifecta":
        # The maximum KRA field is small, so an explicit inverse is clearer and exact.
        inverse = {
            n * (n - 1) * (n - 2): n for n in range(3, 31)
        }
        implied_n = n_by_race.map(inverse)
        expected_rows = implied_n * (implied_n - 1) * (implied_n - 2)
    else:
        raise ValueError(f"unsupported behavioral target market: {market}")
    if bool(expected_rows.isna().any()) or not bool(n_by_race.eq(expected_rows).all()):
        raise ValueError(f"{market} target does not have complete support")
    frame["raw_price"] = 1.0 / frame["odds"].astype(float)
    frame["actual_price_share"] = frame["raw_price"] / frame.groupby("race_id")[
        "raw_price"
    ].transform("sum")
    return frame


def load_pool_levels(data_root: Path, races: pd.DataFrame) -> pd.DataFrame:
    """Load race overrounds used only to fit one out-of-year pool level."""
    race_year = races[["race_id", "year"]]
    rows: list[pd.DataFrame] = []
    for market in TARGET_MARKETS:
        frame = read_parquets(
            data_root,
            market,
            columns=["race_id", "odds", "is_capped_odds"],
        )
        frame = frame.merge(race_year, on="race_id", how="inner", validate="many_to_one")
        frame["raw_price"] = 1.0 / frame["odds"].astype(float)
        grouped = (
            frame.groupby(["race_id", "year"], as_index=False)
            .agg(
                pool_level=("raw_price", "sum"),
                any_capped=("is_capped_odds", "any"),
            )
        )
        grouped["target_market"] = market
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _horse_lookup(validation: pd.DataFrame, temperature: StageTemperature, stage: int) -> pd.DataFrame:
    columns = ["race_id", "horse_no", "objective_probability", "n_valid_horses"]
    result = validation[columns].copy()
    if stage == 1:
        result["stage_score"] = result["objective_probability"]
    else:
        remaining_size = result["n_valid_horses"].to_numpy() - stage + 1
        alpha = temperature.alpha(remaining_size)
        result["stage_score"] = np.power(result["objective_probability"], alpha)
    result["stage_total"] = result.groupby("race_id")["stage_score"].transform("sum")
    return result


def attach_event_probabilities(
    target: pd.DataFrame,
    validation: pd.DataFrame,
    probability_model: str,
    fold: FoldModel,
) -> pd.DataFrame:
    stage2, stage3 = model_temperatures(fold, probability_model)
    stage1_lookup = _horse_lookup(validation, StageTemperature(0.0, 0.0), 1)
    stage2_lookup = _horse_lookup(validation, stage2, 2)
    stage3_lookup = _horse_lookup(validation, stage3, 3)

    result = target.copy()
    first = stage1_lookup.rename(
        columns={
            "horse_no": "first_no",
            "objective_probability": "p1",
            "n_valid_horses": "n_valid_horses",
            "stage_score": "stage1_first_score",
            "stage_total": "stage1_total",
        }
    )
    result = result.merge(first, on=["race_id", "first_no"], how="left", validate="many_to_one")

    stage2_first = stage2_lookup[["race_id", "horse_no", "stage_score"]].rename(
        columns={"horse_no": "first_no", "stage_score": "stage2_first_score"}
    )
    stage2_second = stage2_lookup[["race_id", "horse_no", "stage_score", "stage_total"]].rename(
        columns={
            "horse_no": "second_no",
            "stage_score": "stage2_second_score",
            "stage_total": "stage2_total",
        }
    )
    result = result.merge(stage2_first, on=["race_id", "first_no"], how="left", validate="many_to_one")
    result = result.merge(stage2_second, on=["race_id", "second_no"], how="left", validate="many_to_one")
    result["p2_cond"] = result["stage2_second_score"] / (
        result["stage2_total"] - result["stage2_first_score"]
    )
    result["p_joint"] = result["p1"] * result["p2_cond"]

    if "third_no" in result.columns:
        stage3_first = stage3_lookup[["race_id", "horse_no", "stage_score"]].rename(
            columns={"horse_no": "first_no", "stage_score": "stage3_first_score"}
        )
        stage3_second = stage3_lookup[["race_id", "horse_no", "stage_score"]].rename(
            columns={"horse_no": "second_no", "stage_score": "stage3_second_score"}
        )
        stage3_third = stage3_lookup[["race_id", "horse_no", "stage_score", "stage_total"]].rename(
            columns={
                "horse_no": "third_no",
                "stage_score": "stage3_third_score",
                "stage_total": "stage3_total",
            }
        )
        result = result.merge(stage3_first, on=["race_id", "first_no"], how="left", validate="many_to_one")
        result = result.merge(stage3_second, on=["race_id", "second_no"], how="left", validate="many_to_one")
        result = result.merge(stage3_third, on=["race_id", "third_no"], how="left", validate="many_to_one")
        result["p3_cond"] = result["stage3_third_score"] / (
            result["stage3_total"]
            - result["stage3_first_score"]
            - result["stage3_second_score"]
        )
        result["p_joint"] *= result["p3_cond"]
    probability_columns = ["p1", "p2_cond", "p_joint"]
    if "third_no" in result.columns:
        probability_columns.append("p3_cond")
    if bool(result[probability_columns].isna().any().any()):
        raise ValueError("event-probability merge produced missing values")
    if bool((result[probability_columns] <= 0).any().any()) or bool(
        (result[probability_columns] > 1 + 1e-10).any().any()
    ):
        raise ValueError("event probabilities are outside (0, 1]")
    return result


def tail_predictors(fold: FoldModel) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "isotonic_clip": fold.win_weight.predict,
        "prelec": fold.prelec.predict,
        "power": fold.power.predict,
    }


def price_model_arguments(frame: pd.DataFrame, market: str, model: str) -> list[np.ndarray]:
    if model in ("M-U", "M-R"):
        return [frame["p_joint"].to_numpy()]
    if model == "M-S2" and market == "exacta":
        return [frame["p1"].to_numpy(), frame["p2_cond"].to_numpy()]
    if model == "M-S2" and market == "trifecta":
        return [
            frame["p1"].to_numpy(),
            (frame["p2_cond"] * frame["p3_cond"]).to_numpy(),
        ]
    if model == "M-S3" and market == "trifecta":
        return [
            frame["p1"].to_numpy(),
            frame["p2_cond"].to_numpy(),
            frame["p3_cond"].to_numpy(),
        ]
    raise ValueError(f"unsupported model-market pair: {model}, {market}")


def score_price_model(
    frame: pd.DataFrame,
    market: str,
    model: str,
    predictor: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    arguments = price_model_arguments(frame, market, model)
    score = np.ones(len(frame), dtype=float)
    for argument in arguments:
        score *= predictor(argument)
    if np.any(~np.isfinite(score)) or np.any(score <= 0):
        raise ValueError("behavioral price model produced invalid scores")
    return score, arguments


def race_metric_rows(
    frame: pd.DataFrame,
    score: np.ndarray,
    arguments: list[np.ndarray],
    fold: FoldModel,
    probability_model: str,
    tail_model: str,
    price_model: str,
    target_market: str,
    pool_level: float,
) -> pd.DataFrame:
    work = frame[["race_id", "actual_price_share", "raw_price"]].copy()
    work["score"] = score
    work["predicted_price_share"] = work["score"] / work.groupby("race_id")[
        "score"
    ].transform("sum")
    if not np.isfinite(pool_level) or pool_level <= 0:
        raise ValueError("training-pool level must be finite and positive")
    work["predicted_raw_price"] = work["predicted_price_share"] * pool_level

    lower, upper = fold.win_weight.support
    supported = np.ones(len(work), dtype=bool)
    for argument in arguments:
        supported &= (argument >= lower) & (argument <= upper)
    work["supported"] = supported.astype(float)
    actual = work["actual_price_share"].to_numpy()
    predicted = work["predicted_price_share"].to_numpy()
    midpoint = 0.5 * (actual + predicted)
    work["abs"] = np.abs(actual - predicted)
    work["log_sq"] = np.square(np.log(np.clip(predicted, EPSILON, None)) - np.log(actual))
    work["js_term"] = 0.5 * (
        actual * np.log(actual / midpoint) + predicted * np.log(predicted / midpoint)
    )
    work["raw_abs"] = np.abs(work["raw_price"] - work["predicted_raw_price"])
    grouped = work.groupby("race_id", sort=False)
    result = grouped.agg(
        tv=("abs", lambda values: 0.5 * float(values.sum())),
        mae=("abs", "mean"),
        mean_log_sq=("log_sq", "mean"),
        js=("js_term", "sum"),
        raw_mae=("raw_abs", "mean"),
        support_share=("supported", "mean"),
        n_combinations=("score", "size"),
    ).reset_index()
    result["log_rmse"] = np.sqrt(result.pop("mean_log_sq"))
    result["validation_year"] = fold.validation_year
    result["probability_model"] = probability_model
    result["tail_model"] = tail_model
    result["price_model"] = price_model
    result["target_market"] = target_market
    result["training_pool_level"] = pool_level
    return result


def behavioral_price_metrics(
    target: pd.DataFrame,
    validation: pd.DataFrame,
    fold: FoldModel,
    market: str,
    pool_level: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for probability_model in PROBABILITY_MODELS:
        events = attach_event_probabilities(target, validation, probability_model, fold)
        for tail_model, predictor in tail_predictors(fold).items():
            for price_model in PRICE_MODELS[market]:
                score, arguments = score_price_model(
                    events, market, price_model, predictor
                )
                rows.append(
                    race_metric_rows(
                        events,
                        score,
                        arguments,
                        fold,
                        probability_model,
                        tail_model,
                        price_model,
                        market,
                        pool_level,
                    )
                )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def parameter_row(
    fold: FoldModel,
    train: pd.DataFrame,
    pool_levels: dict[str, float],
) -> dict[str, object]:
    row = {
        "validation_year": fold.validation_year,
        "n_training_races": int(train["race_id"].nunique()),
        "n_training_horses": int(len(train)),
        "stage2_intercept": fold.stage2.intercept,
        "stage2_field_slope": fold.stage2.field_slope,
        "stage2_alpha_n10": float(fold.stage2.alpha(10)),
        "stage3_intercept": fold.stage3.intercept,
        "stage3_field_slope": fold.stage3.field_slope,
        "stage3_alpha_n10": float(fold.stage3.alpha(10)),
        "win_probability_support_min": fold.win_weight.support[0],
        "win_probability_support_max": fold.win_weight.support[1],
        "prelec_alpha": fold.prelec.alpha,
        "prelec_beta": fold.prelec.beta,
        "power_exponent": fold.power.exponent,
    }
    row.update(
        {f"{market}_training_pool_level": value for market, value in pool_levels.items()}
    )
    return row


def rank_validation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (probability_model, stage), group in frame.groupby(
        ["probability_model", "stage"], sort=True
    ):
        weights = group["n_races"].to_numpy(dtype=float)
        rows.append(
            {
                "probability_model": probability_model,
                "stage": int(stage),
                "n_validation_races": int(weights.sum()),
                "n_validation_years": int(group["validation_year"].nunique()),
                "mean_log_loss": float(
                    np.average(group["mean_log_loss"], weights=weights)
                ),
                "mean_brier": float(
                    np.average(group["mean_brier"], weights=weights)
                ),
                # Fold ECE is not linearly aggregable; report the race-weighted
                # mean of the eight independently evaluated fold ECE values.
                "mean_fold_ece_10": float(
                    np.average(group["ece_10"], weights=weights)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stage", "probability_model"]
    ).reset_index(drop=True)


def price_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["probability_model", "tail_model", "price_model", "target_market"]
    return (
        frame.groupby(keys, as_index=False)
        .agg(
            n_races=("race_id", "nunique"),
            n_years=("validation_year", "nunique"),
            median_tv=("tv", "median"),
            mean_tv=("tv", "mean"),
            median_mae=("mae", "median"),
            median_log_rmse=("log_rmse", "median"),
            median_js=("js", "median"),
            median_raw_mae=("raw_mae", "median"),
            mean_support_share=("support_share", "mean"),
            all_arguments_supported_race_share=(
                "support_share",
                lambda values: float(values.ge(1.0 - 1e-12).mean()),
            ),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"20260810|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def bootstrap_median_interval(
    values: np.ndarray,
    label: str,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("bootstrap input must be a non-empty finite vector")
    rng = np.random.default_rng(_stable_seed(label))
    estimates: list[np.ndarray] = []
    batch_size = 40
    for start in range(0, reps, batch_size):
        count = min(batch_size, reps - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates.append(np.median(values[indices], axis=1))
    distribution = np.concatenate(estimates)
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    return float(lower), float(upper)


def paired_model_improvements(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired race-level loss reductions for the directly identified contrasts."""
    primary = frame[frame["probability_model"].eq("stage_temperature")].copy()
    comparisons = {
        "exacta": (("M-R", "M-S2"),),
        "trifecta": (
            ("M-R", "M-S2"),
            ("M-R", "M-S3"),
            ("M-S2", "M-S3"),
        ),
    }
    rows: list[dict[str, object]] = []
    for (tail_model, target_market), group in primary.groupby(
        ["tail_model", "target_market"], sort=True
    ):
        for baseline, challenger in comparisons[target_market]:
            for metric in ("tv", "log_rmse", "js"):
                pivot = group.pivot(
                    index=["validation_year", "race_id"],
                    columns="price_model",
                    values=metric,
                )
                paired = pivot[[baseline, challenger]].dropna()
                difference = (paired[baseline] - paired[challenger]).to_numpy()
                label = f"{tail_model}|{target_market}|{baseline}|{challenger}|{metric}"
                ci_lower, ci_upper = bootstrap_median_interval(difference, label)
                by_year = (
                    pd.Series(difference, index=paired.index)
                    .groupby(level="validation_year")
                    .median()
                )
                rows.append(
                    {
                        "probability_model": "stage_temperature",
                        "tail_model": tail_model,
                        "target_market": target_market,
                        "baseline_model": baseline,
                        "challenger_model": challenger,
                        "loss": metric,
                        "n_races": int(len(paired)),
                        "n_years": int(len(by_year)),
                        "median_loss_reduction": float(np.median(difference)),
                        "mean_loss_reduction": float(np.mean(difference)),
                        "bootstrap_median_ci_lower": ci_lower,
                        "bootstrap_median_ci_upper": ci_upper,
                        "challenger_better_race_share": float(np.mean(difference > 0)),
                        "years_with_positive_median": int((by_year > 0).sum()),
                        "minimum_year_median_reduction": float(by_year.min()),
                        "maximum_year_median_reduction": float(by_year.max()),
                    }
                )
    result = pd.DataFrame(rows)
    power_rows = result[result["tail_model"].eq("power")]
    numeric = power_rows[
        ["median_loss_reduction", "mean_loss_reduction"]
    ].to_numpy()
    if len(power_rows) and not np.allclose(numeric, 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(
            "power weighting must make reduced and sequential models observationally equivalent"
        )
    return result.sort_values(
        [
            "target_market",
            "tail_model",
            "baseline_model",
            "challenger_model",
            "loss",
        ]
    ).reset_index(drop=True)


def write_manifest(
    output_dir: Path,
    races: pd.DataFrame,
    rank_rows: pd.DataFrame,
    price_rows: pd.DataFrame,
) -> None:
    manifest = {
        "schema_version": 1,
        "date_scope": {
            "start": START_DATE,
            "end": END_DATE,
            "excluded_years": sorted(EXCLUDED_YEARS),
            "excluded_dates": sorted(EXCLUDED_DATES),
        },
        "common_support_quantiles": list(COMMON_SUPPORT_QUANTILES),
        "n_rank_races": int(races["race_id"].nunique()),
        "validation_years": sorted(int(value) for value in races["year"].unique()),
        "probability_models": list(PROBABILITY_MODELS),
        "tail_models": list(TAIL_MODELS),
        "price_models": {key: list(value) for key, value in PRICE_MODELS.items()},
        "identification_note": (
            "M-U and M-R are observationally equivalent when nonparametrically "
            "recovered from the same win-price schedule; the direct contrast is "
            "M-R versus M-S2/M-S3."
        ),
        "rank_validation_rows": int(len(rank_rows)),
        "price_metric_rows": int(len(price_rows)),
        "uncapped_price_races": {
            market: int(price_rows.loc[price_rows["target_market"].eq(market), "race_id"].nunique())
            if not price_rows.empty
            else 0
            for market in TARGET_MARKETS
        },
    }
    (output_dir / "behavioral_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    races, horses = load_horse_panel(args.data_root, args.max_races)
    years = sorted(int(value) for value in races["year"].unique())
    if len(years) < 3:
        raise ValueError("year-cross-fitting requires at least three years")

    rank_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    price_rows: list[pd.DataFrame] = []
    pool_level_frame = (
        pd.DataFrame()
        if args.skip_price_models
        else load_pool_levels(args.data_root, races)
    )
    for year in years:
        train = horses.loc[~horses["year"].eq(year)].copy()
        validation_raw = horses.loc[horses["year"].eq(year)].copy()
        fold, calibrated_train = fit_fold(train, year)
        validation = apply_calibration(validation_raw, fold.calibration)
        rank_rows.append(rank_validation(validation, fold))
        fold_pool_levels = {
            market: float(
                pool_level_frame.loc[
                    pool_level_frame["target_market"].eq(market)
                    & pool_level_frame["year"].ne(year)
                    & ~pool_level_frame["any_capped"],
                    "pool_level",
                ].median()
            )
            for market in TARGET_MARKETS
        } if not args.skip_price_models else {}
        parameter_rows.append(
            parameter_row(fold, calibrated_train, fold_pool_levels)
        )
        print(
            f"fold {year}: train={train['race_id'].nunique():,}, "
            f"validation={validation['race_id'].nunique():,}, "
            f"alpha2(n=10)={fold.stage2.alpha(10):.4f}, "
            f"alpha3(n=10)={fold.stage3.alpha(10):.4f}",
            flush=True,
        )
        if args.skip_price_models:
            continue
        validation_ids = set(validation["race_id"])
        for market in TARGET_MARKETS:
            target = uncapped_target(args.data_root, market, year, validation_ids)
            if target.empty:
                continue
            metrics = behavioral_price_metrics(
                target,
                validation,
                fold,
                market,
                fold_pool_levels[market],
            )
            price_rows.append(metrics)
            print(
                f"  {market}: {target['race_id'].nunique():,} uncapped races, "
                f"{len(target):,} combinations",
                flush=True,
            )

    rank_frame = pd.concat(rank_rows, ignore_index=True)
    rank_frame.to_csv(args.output_dir / "rank_probability_validation_by_year.csv", index=False)
    rank_validation_summary(rank_frame).to_csv(
        args.output_dir / "rank_probability_validation.csv", index=False
    )
    pd.DataFrame(parameter_rows).to_csv(
        args.output_dir / "behavioral_model_parameters.csv", index=False
    )

    if price_rows:
        price_frame = pd.concat(price_rows, ignore_index=True)
        price_frame.to_csv(
            args.output_dir / "behavioral_model_metrics_by_race.csv", index=False
        )
        price_summary(price_frame).to_csv(
            args.output_dir / "behavioral_model_comparison.csv", index=False
        )
        paired_model_improvements(price_frame).to_csv(
            args.output_dir / "behavioral_model_improvements.csv", index=False
        )
    else:
        price_frame = pd.DataFrame()
    write_manifest(args.output_dir, races, rank_frame, price_frame)


if __name__ == "__main__":
    main()
