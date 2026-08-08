#!/usr/bin/env python3
"""Implement the co-primary cross-pool price-coherence analysis.

Panel A compares point prices on the common uncapped sample. Panel B preserves
published rounding/censoring information as normalized interval price sets and
computes exact lower TV bounds plus certified outer upper bounds. Benchmark
assignments, permutation draws, bootstrap resampling, and numeric tolerances are
fixed for deterministic reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from analysis.data_audit import (
    MARKET_SPECS,
    TARGET_MARKETS,
    parse_horse_list,
    prepare_races,
    read_parquets,
)

SOURCE_MARKET = "trifecta"
MODELS = ("main", "harville", "permutation", "other_race", "uniform")
RANDOM_SEED = 20260809
BOOTSTRAP_REPS = 999
ROUNDING_HALF_WIDTH = 0.05
DISPLAY_CAP = 9999.9
EPSILON = 1e-12
TV_THRESHOLD = 0.05
TV_SENSITIVITY_THRESHOLDS = (0.025, 0.10)
LP_TOL = 1e-9


@dataclass(frozen=True)
class PriceSet:
    """Raw inverse-odds intervals whose normalization defines a price set."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.ndim != 1 or upper.ndim != 1 or len(lower) != len(upper):
            raise ValueError("price-set bounds must be same-length vectors")
        if np.any(lower < -LP_TOL) or np.any(upper <= 0) or np.any(lower - upper > LP_TOL):
            raise ValueError("invalid price-set interval")
        if float(np.sum(upper)) <= 0:
            raise ValueError("price set has zero total upper mass")
        object.__setattr__(self, "lower", np.maximum(lower, 0.0))
        object.__setattr__(self, "upper", upper)

    @property
    def size(self) -> int:
        return len(self.lower)


@dataclass(frozen=True)
class RaceSlices:
    frame: pd.DataFrame
    slices: Mapping[str, tuple[int, int]]

    def get(self, race_id: str) -> pd.DataFrame:
        start, count = self.slices[race_id]
        return self.frame.iloc[start : start + count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("KRA/parsed"))
    parser.add_argument("--sample-csv", type=Path, default=Path("outputs/analysis_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))
    parser.add_argument("--max-races", type=int, default=0, help="deterministic smoke-test limit")
    parser.add_argument("--skip-bounds", action="store_true", help="Panel A only; never use for final results")
    return parser.parse_args()


def stable_uint(text: str) -> int:
    digest = hashlib.sha256(f"{RANDOM_SEED}|{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def normalize_inverse_odds(odds: np.ndarray) -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(odds)) or np.any(odds <= 0):
        raise ValueError("odds must be finite and positive")
    raw = 1.0 / odds
    total = float(raw.sum())
    if total <= 0:
        raise ValueError("inverse odds sum must be positive")
    return raw / total


def odds_to_price_set(odds: np.ndarray, capped: np.ndarray) -> PriceSet:
    """Translate displayed total-payout multiples into reciprocal-price intervals.

    Non-capped KRA displays are recorded to one decimal place; the maintained
    design treats them as rounded-to-nearest-tenth observations, hence +/-0.05.
    A capped 9999.9 display gives only a lower bound on the true payout multiple,
    so its reciprocal-price lower bound is zero.
    """
    odds = np.asarray(odds, dtype=float)
    capped = np.asarray(capped, dtype=bool)
    if len(odds) != len(capped):
        raise ValueError("odds and cap vectors differ in length")
    if np.any(~np.isfinite(odds)) or np.any(odds <= 0):
        raise ValueError("displayed odds must be finite and positive")
    d_lower = np.maximum(odds - ROUNDING_HALF_WIDTH, EPSILON)
    d_upper = odds + ROUNDING_HALF_WIDTH
    d_lower[capped] = np.maximum(odds[capped], DISPLAY_CAP)
    d_upper[capped] = np.inf
    lower = np.zeros_like(odds, dtype=float)
    finite = np.isfinite(d_upper)
    lower[finite] = 1.0 / d_upper[finite]
    upper = 1.0 / d_lower
    return PriceSet(lower, upper)


def point_price_set(probability: np.ndarray) -> PriceSet:
    probability = np.asarray(probability, dtype=float)
    if np.any(probability < 0):
        raise ValueError("negative probability")
    total = float(probability.sum())
    if total <= 0:
        raise ValueError("point distribution has zero mass")
    p = probability / total
    return PriceSet(p.copy(), p.copy())


def aggregate_price_set(source: PriceSet, group_index: np.ndarray, n_groups: int) -> PriceSet:
    group_index = np.asarray(group_index, dtype=np.int64)
    if len(group_index) != source.size:
        raise ValueError("group map length differs from source price set")
    if np.any(group_index < 0) or np.any(group_index >= n_groups):
        raise ValueError("invalid group index")
    lower = np.bincount(group_index, weights=source.lower, minlength=n_groups)
    upper = np.bincount(group_index, weights=source.upper, minlength=n_groups)
    return PriceSet(lower, upper)


def price_set_component_bounds(price_set: PriceSet) -> tuple[np.ndarray, np.ndarray]:
    """Exact componentwise extrema of a normalized independent interval box."""
    l = price_set.lower
    u = price_set.upper
    sum_l = float(l.sum())
    sum_u = float(u.sum())
    min_den = l + (sum_u - u)
    max_den = u + (sum_l - l)
    p_min = np.divide(l, min_den, out=np.zeros_like(l), where=min_den > 0)
    p_max = np.divide(u, max_den, out=np.ones_like(u), where=max_den > 0)
    return np.clip(p_min, 0.0, 1.0), np.clip(p_max, 0.0, 1.0)


def tv_upper_outer(left: PriceSet, right: PriceSet) -> float:
    """Certified outer upper bound from exact componentwise difference ranges."""
    if left.size != right.size:
        raise ValueError("TV price sets must have the same dimension")
    lmin, lmax = price_set_component_bounds(left)
    rmin, rmax = price_set_component_bounds(right)
    diff_min = lmin - rmax
    diff_max = lmax - rmin
    bound = 0.5 * float(np.maximum(np.abs(diff_min), np.abs(diff_max)).sum())
    return min(1.0, max(0.0, bound))


def _scale_bounds(price_set: PriceSet) -> tuple[float, float | None]:
    sum_u = float(price_set.upper.sum())
    sum_l = float(price_set.lower.sum())
    lower = 1.0 / sum_u
    upper = None if sum_l <= 0 else 1.0 / sum_l
    return lower, upper


def tv_lower_exact(left: PriceSet, right: PriceSet) -> float:
    """Exact minimum TV distance between two normalized interval price sets."""
    if left.size != right.size:
        raise ValueError("TV price sets must have the same dimension")
    cdim = left.size
    p0 = 0
    q0 = cdim
    sp_idx = 2 * cdim
    sq_idx = 2 * cdim + 1
    z0 = 2 * cdim + 2
    nvar = 3 * cdim + 2

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_ub = np.zeros(6 * cdim, dtype=float)
    row = 0
    for i in range(cdim):
        rows += [row, row]
        cols += [p0 + i, sp_idx]
        data += [1.0, -float(left.upper[i])]
        row += 1
        rows += [row, row]
        cols += [p0 + i, sp_idx]
        data += [-1.0, float(left.lower[i])]
        row += 1
        rows += [row, row]
        cols += [q0 + i, sq_idx]
        data += [1.0, -float(right.upper[i])]
        row += 1
        rows += [row, row]
        cols += [q0 + i, sq_idx]
        data += [-1.0, float(right.lower[i])]
        row += 1
        rows += [row, row, row]
        cols += [p0 + i, q0 + i, z0 + i]
        data += [1.0, -1.0, -1.0]
        row += 1
        rows += [row, row, row]
        cols += [q0 + i, p0 + i, z0 + i]
        data += [1.0, -1.0, -1.0]
        row += 1
    a_ub = sparse.coo_matrix((data, (rows, cols)), shape=(6 * cdim, nvar)).tocsr()

    eq_rows = np.concatenate([np.zeros(cdim, dtype=int), np.ones(cdim, dtype=int)])
    eq_cols = np.concatenate([np.arange(p0, p0 + cdim), np.arange(q0, q0 + cdim)])
    eq_data = np.ones(2 * cdim, dtype=float)
    a_eq = sparse.coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(2, nvar)).tocsr()
    b_eq = np.ones(2, dtype=float)

    objective = np.zeros(nvar, dtype=float)
    objective[z0:] = 0.5
    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * nvar
    bounds[sp_idx] = _scale_bounds(left)
    bounds[sq_idx] = _scale_bounds(right)

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={
            "dual_feasibility_tolerance": LP_TOL,
            "primal_feasibility_tolerance": LP_TOL,
        },
    )
    if not result.success:
        raise RuntimeError(f"TV lower-bound LP failed: {result.message}")
    return min(1.0, max(0.0, float(result.fun)))


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    actual = actual / actual.sum()
    predicted = predicted / predicted.sum()
    diff = actual - predicted
    tv = 0.5 * float(np.abs(diff).sum())
    mae = float(np.abs(diff).mean())
    midpoint = 0.5 * (actual + predicted)
    js_terms_a = np.where(actual > 0, actual * np.log(actual / midpoint), 0.0)
    js_terms_p = np.where(predicted > 0, predicted * np.log(predicted / midpoint), 0.0)
    js = 0.5 * float(js_terms_a.sum() + js_terms_p.sum())
    log_a = np.log(actual + EPSILON)
    log_p = np.log(predicted + EPSILON)
    rmsle = float(np.sqrt(np.mean((log_a - log_p) ** 2)))
    clr_a = log_a - log_a.mean()
    clr_p = log_p - log_p.mean()
    clr_distance = float(np.sqrt(np.mean((clr_a - clr_p) ** 2)))
    centered = actual - actual.mean()
    sst = float(np.dot(centered, centered))
    sse = float(np.dot(diff, diff))
    r2 = float("nan") if sst <= 0 else 1.0 - sse / sst
    if len(actual) < 2 or np.std(actual) <= 0 or np.std(predicted) <= 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(actual, predicted)[0, 1])
    denom = float(np.linalg.norm(actual) * np.linalg.norm(predicted))
    cosine = float("nan") if denom <= 0 else float(np.dot(actual, predicted) / denom)
    return {
        "tv": tv,
        "mae": mae,
        "js": js,
        "rmsle": rmsle,
        "clr_distance": clr_distance,
        "r2": r2,
        "correlation": corr,
        "cosine": cosine,
    }


def market_sort_columns(market: str) -> list[str]:
    return ["race_id", *MARKET_SPECS[market].keys]


def load_market(data_root: Path, market: str, race_ids: set[str]) -> RaceSlices:
    spec = MARKET_SPECS[market]
    columns = ["race_id", *spec.keys, "odds", "is_capped_odds"]
    frame = read_parquets(data_root, market, columns=columns)
    frame = frame[frame["race_id"].isin(race_ids)].copy()
    frame = frame.sort_values(market_sort_columns(market)).reset_index(drop=True)
    ids = frame["race_id"].to_numpy()
    unique, starts, counts = np.unique(ids, return_index=True, return_counts=True)
    slices = {
        str(race_id): (int(start), int(count))
        for race_id, start, count in zip(unique, starts, counts)
    }
    missing = race_ids.difference(slices)
    if missing:
        raise ValueError(f"{market}: missing {len(missing)} eligible races")
    return RaceSlices(frame=frame, slices=slices)


def target_key(row: tuple[int, int, int], target: str) -> object:
    first, second, third = row
    if target == "win":
        return first
    if target == "exacta":
        return (first, second)
    if target == "quinella":
        return tuple(sorted((first, second)))
    if target == "trio":
        return tuple(sorted((first, second, third)))
    raise KeyError(target)


def target_keys_from_frame(frame: pd.DataFrame, target: str) -> list[object]:
    spec = MARKET_SPECS[target]
    if target == "win":
        return [int(v) for v in frame[spec.keys[0]].to_numpy()]
    return [
        tuple(int(v) for v in values)
        for values in frame[list(spec.keys)].itertuples(index=False, name=None)
    ]


def source_group_index(source: pd.DataFrame, target: pd.DataFrame, target_name: str) -> np.ndarray:
    target_keys = target_keys_from_frame(target, target_name)
    mapping = {key: idx for idx, key in enumerate(target_keys)}
    triples = source[["first_no", "second_no", "third_no"]].itertuples(index=False, name=None)
    try:
        groups = np.fromiter(
            (
                mapping[target_key(tuple(map(int, triple)), target_name)]
                for triple in triples
            ),
            dtype=np.int64,
            count=len(source),
        )
    except KeyError as exc:
        raise ValueError(f"source state does not map into {target_name} support: {exc}") from exc
    if len(np.unique(groups)) != len(target_keys):
        raise ValueError(f"source marginalization does not cover every {target_name} outcome")
    return groups


def harville_trifecta(source: pd.DataFrame, win: pd.DataFrame) -> np.ndarray:
    win_keys = [int(v) for v in win["horse_no"].to_numpy()]
    p = normalize_inverse_odds(win["odds"].to_numpy(dtype=float))
    pmap = dict(zip(win_keys, p))
    values = np.empty(len(source), dtype=float)
    for idx, (first, second, third) in enumerate(
        source[["first_no", "second_no", "third_no"]].itertuples(index=False, name=None)
    ):
        pi = pmap[int(first)]
        pj = pmap[int(second)]
        pk = pmap[int(third)]
        denom2 = 1.0 - pi
        denom3 = 1.0 - pi - pj
        if denom2 <= 0 or denom3 <= 0:
            raise ValueError("invalid Harville denominator")
        values[idx] = pi * (pj / denom2) * (pk / denom3)
    values = np.maximum(values, 0.0)
    return values / values.sum()


def aggregate_point(point: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    value = np.bincount(
        groups, weights=np.asarray(point, dtype=float), minlength=n_groups
    )
    return value / value.sum()


def choose_other_race(race_id: str, peers: list[str], label: str) -> str:
    if len(peers) < 2:
        raise ValueError(f"no alternate donor for {race_id}")
    idx = stable_uint(f"donor|{label}|{race_id}") % len(peers)
    if peers[idx] == race_id:
        idx = (idx + 1) % len(peers)
    return peers[idx]


def deterministic_permutation(length: int, race_id: str, target: str, panel: str) -> np.ndarray:
    rng = np.random.default_rng(stable_uint(f"perm|{panel}|{target}|{race_id}"))
    return rng.permutation(length)


def grouped_ids_by_field(races: pd.DataFrame, race_ids: Iterable[str]) -> dict[int, list[str]]:
    subset = races[races["race_id"].isin(set(race_ids))][
        ["race_id", "n_valid_horses"]
    ]
    result: dict[int, list[str]] = {}
    for n, group in subset.groupby("n_valid_horses"):
        result[int(n)] = sorted(group["race_id"].astype(str).tolist())
    return result


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


def read_frozen_sample(sample_csv: Path) -> pd.DataFrame:
    if not sample_csv.exists():
        raise FileNotFoundError(
            f"{sample_csv} is missing; run `python -m analysis.data_audit --strict` first"
        )
    sample = pd.read_csv(sample_csv)
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


def target_race_ids(sample: pd.DataFrame, target: str, column: str) -> list[str]:
    mask = _boolean_series(sample[column])
    rows = sample[sample["target_market"].eq(target) & mask]
    return sorted(rows["race_id"].astype(str).tolist())


def common_race_ids(sample: pd.DataFrame, column: str) -> list[str]:
    sets = [set(target_race_ids(sample, target, column)) for target in TARGET_MARKETS]
    if not all(s == sets[0] for s in sets[1:]):
        raise ValueError(f"target samples differ for {column}")
    return sorted(sets[0])


def race_metadata(data_root: Path, race_ids: set[str]) -> pd.DataFrame:
    races = prepare_races(data_root)
    races = races[races["race_id"].isin(race_ids)].copy()
    races["year"] = pd.to_datetime(races["race_date"]).dt.year
    races["valid_horse_tuple"] = races["valid_horses"].map(parse_horse_list)
    return races


def panel_a(
    races: pd.DataFrame,
    trifecta: RaceSlices,
    win: RaceSlices,
    target_name: str,
    target: RaceSlices,
    clean_ids: list[str],
    clean_peers: dict[int, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_map = races.set_index("race_id")["n_valid_horses"].astype(int).to_dict()
    for race_id in clean_ids:
        source = trifecta.get(race_id)
        actual_frame = target.get(race_id)
        win_frame = win.get(race_id)
        groups = source_group_index(source, actual_frame, target_name)
        cdim = len(actual_frame)
        q_main = normalize_inverse_odds(source["odds"].to_numpy(dtype=float))
        actual = normalize_inverse_odds(actual_frame["odds"].to_numpy(dtype=float))
        q_h = harville_trifecta(source, win_frame)
        q_uniform = np.full(len(source), 1.0 / len(source), dtype=float)
        permutation = deterministic_permutation(len(source), race_id, target_name, "A")
        q_perm = q_main[permutation]
        n = int(n_map[race_id])
        donor_id = choose_other_race(race_id, clean_peers[n], "A")
        donor_source = trifecta.get(donor_id)
        if len(donor_source) != len(source):
            raise ValueError("same-field donor has different trifecta support size")
        q_donor = normalize_inverse_odds(donor_source["odds"].to_numpy(dtype=float))
        predictions = {
            "main": aggregate_point(q_main, groups, cdim),
            "harville": aggregate_point(q_h, groups, cdim),
            "permutation": aggregate_point(q_perm, groups, cdim),
            "other_race": aggregate_point(q_donor, groups, cdim),
            "uniform": aggregate_point(q_uniform, groups, cdim),
        }
        for model, predicted in predictions.items():
            rec: dict[str, object] = {
                "panel": "A",
                "race_id": race_id,
                "target_market": target_name,
                "model": model,
                "n_valid_horses": n,
                "n_outcomes": cdim,
                "donor_race_id": donor_id if model == "other_race" else "",
            }
            rec.update(point_metrics(actual, predicted))
            rows.append(rec)
    return pd.DataFrame(rows)


def interval_for_frame(frame: pd.DataFrame) -> PriceSet:
    return odds_to_price_set(
        frame["odds"].to_numpy(dtype=float),
        frame["is_capped_odds"].fillna(False).to_numpy(dtype=bool),
    )


def panel_b(
    races: pd.DataFrame,
    trifecta: RaceSlices,
    win: RaceSlices,
    target_name: str,
    target: RaceSlices,
    full_ids: list[str],
    full_peers: dict[int, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_map = races.set_index("race_id")["n_valid_horses"].astype(int).to_dict()
    for race_id in full_ids:
        source = trifecta.get(race_id)
        actual_frame = target.get(race_id)
        win_frame = win.get(race_id)
        groups = source_group_index(source, actual_frame, target_name)
        cdim = len(actual_frame)
        actual_set = interval_for_frame(actual_frame)
        source_set = interval_for_frame(source)
        main_set = aggregate_price_set(source_set, groups, cdim)

        q_h = harville_trifecta(source, win_frame)
        harville_set = point_price_set(aggregate_point(q_h, groups, cdim))
        uniform_set = point_price_set(np.full(cdim, 1.0 / cdim, dtype=float))

        permutation = deterministic_permutation(len(source), race_id, target_name, "B")
        perm_source = PriceSet(
            source_set.lower[permutation], source_set.upper[permutation]
        )
        permutation_set = aggregate_price_set(perm_source, groups, cdim)

        n = int(n_map[race_id])
        donor_id = choose_other_race(race_id, full_peers[n], "B")
        donor_source = trifecta.get(donor_id)
        if len(donor_source) != len(source):
            raise ValueError("same-field donor has different trifecta support size")
        donor_raw_set = interval_for_frame(donor_source)
        donor_set = aggregate_price_set(donor_raw_set, groups, cdim)

        models = {
            "main": main_set,
            "harville": harville_set,
            "permutation": permutation_set,
            "other_race": donor_set,
            "uniform": uniform_set,
        }
        for model, prediction_set in models.items():
            lower = tv_lower_exact(actual_set, prediction_set)
            upper = tv_upper_outer(actual_set, prediction_set)
            if lower - upper > 1e-7:
                raise RuntimeError(
                    f"TV bounds inverted for {race_id} {target_name} {model}"
                )
            rows.append(
                {
                    "panel": "B",
                    "race_id": race_id,
                    "target_market": target_name,
                    "model": model,
                    "n_valid_horses": n,
                    "n_outcomes": cdim,
                    "donor_race_id": donor_id if model == "other_race" else "",
                    "tv_lower": lower,
                    "tv_upper_outer": upper,
                    "mae_lower": 2.0 * lower / cdim,
                    "mae_upper_outer": 2.0 * upper / cdim,
                }
            )
    return pd.DataFrame(rows)


def summarize_panel_a(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, model), group in metrics.groupby(
        ["target_market", "model"], sort=True
    ):
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
    for (target, model), group in bounds.groupby(
        ["target_market", "model"], sort=True
    ):
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


def benchmark_improvements_a(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(
        index=["race_id", "target_market"], columns="model", values="tv"
    )
    rows: list[dict[str, object]] = []
    for target in TARGET_MARKETS:
        sub = pivot.xs(target, level="target_market")
        for benchmark in MODELS[1:]:
            delta = (sub[benchmark] - sub["main"]).to_numpy(dtype=float)
            med, lo, hi = bootstrap_ci(
                delta, label=f"A|improvement|{target}|{benchmark}"
            )
            rows.append(
                {
                    "panel": "A",
                    "target_market": target,
                    "benchmark": benchmark,
                    "n_races": len(delta),
                    "median_improvement_lower": med,
                    "ci_low": lo,
                    "ci_high": hi,
                    "main_better": bool(lo > 0),
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
        low_sub = lower.xs(target, level="target_market")
        up_sub = upper.xs(target, level="target_market")
        for benchmark in MODELS[1:]:
            delta = (low_sub[benchmark] - up_sub["main"]).to_numpy(dtype=float)
            med, lo, hi = bootstrap_ci(
                delta, label=f"B|improvement|{target}|{benchmark}"
            )
            rows.append(
                {
                    "panel": "B",
                    "target_market": target,
                    "benchmark": benchmark,
                    "n_races": len(delta),
                    "median_improvement_lower": med,
                    "ci_low": lo,
                    "ci_high": hi,
                    "main_better": bool(lo > 0),
                    "share_races_robust_main_better": float(np.mean(delta > 0)),
                }
            )
    return pd.DataFrame(rows)


def order_information_test(metrics: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for measure in ("tv", "js"):
        main = metrics[metrics["model"].eq("main")].pivot(
            index="race_id", columns="target_market", values=measure
        )
        harv = metrics[metrics["model"].eq("harville")].pivot(
            index="race_id", columns="target_market", values=measure
        )
        common = main.index.intersection(harv.index)
        exacta_gain = harv.loc[common, "exacta"] - main.loc[common, "exacta"]
        quinella_gain = harv.loc[common, "quinella"] - main.loc[common, "quinella"]
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
    a = summary_a[summary_a["model"].eq("main")].set_index("target_market")
    b = (
        None
        if summary_b is None
        else summary_b[summary_b["model"].eq("main")].set_index("target_market")
    )
    for target in TARGET_MARKETS:
        for threshold in (TV_THRESHOLD, *TV_SENSITIVITY_THRESHOLDS):
            a_pass = bool(a.loc[target, "median_tv_ci_high"] < threshold)
            b_pass = (
                False
                if b is None
                else bool(b.loc[target, "median_tv_upper_outer_ci_high"] < threshold)
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
    races: pd.DataFrame, clean_ids: set[str], full_ids: set[str]
) -> pd.DataFrame:
    subset = races[races["race_id"].isin(full_ids)].copy()
    subset["sample_group"] = np.where(
        subset["race_id"].isin(clean_ids), "clean", "capped"
    )
    rows: list[dict[str, object]] = []
    for group, frame in subset.groupby("sample_group"):
        n = frame["n_valid_horses"].astype(float)
        rows.append(
            {
                "sample_group": group,
                "n_races": len(frame),
                "field_size_median": float(n.median()),
                "field_size_q25": float(n.quantile(0.25)),
                "field_size_q75": float(n.quantile(0.75)),
                "first_year": int(frame["year"].min()),
                "last_year": int(frame["year"].max()),
                "n_meets": int(frame["meet"].nunique()),
            }
        )
    return pd.DataFrame(rows)


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
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
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
    (table_dir / "main_panel_a.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

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
                f"{fmt(row['median_tv_lower'])} & "
                f"[{fmt(row['median_tv_lower_ci_low'])}, {fmt(row['median_tv_lower_ci_high'])}] & "
                f"{fmt(row['median_tv_upper_outer'])} & "
                f"[{fmt(row['median_tv_upper_outer_ci_low'])}, {fmt(row['median_tv_upper_outer_ci_high'])}] \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        (table_dir / "main_panel_b.tex").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

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
    (table_dir / "main_benchmark_comparison.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

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
    (table_dir / "main_order_information.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    output_dir: Path, generated: list[Path], *, max_races: int, bounds: bool
) -> None:
    payload = {
        "schema_version": 1,
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


def main() -> None:
    args = parse_args()
    sample = read_frozen_sample(args.sample_csv)
    clean_ids = common_race_ids(sample, "eligible_clean_point_sample")
    full_ids = common_race_ids(sample, "eligible_complete_sample")
    if args.max_races > 0:
        clean_ids = clean_ids[: args.max_races]
        full_ids = full_ids[: args.max_races]
        full_ids = sorted(set(full_ids).union(clean_ids))
    if not clean_ids or not full_ids:
        raise ValueError("analysis sample is empty")

    all_ids = set(full_ids)
    races = race_metadata(args.data_root, all_ids)
    if len(races) != len(all_ids):
        raise ValueError("race metadata does not cover the analysis sample")
    trifecta = load_market(args.data_root, SOURCE_MARKET, all_ids)
    win = load_market(args.data_root, "win", all_ids)
    clean_peers = grouped_ids_by_field(races, clean_ids)
    full_peers = grouped_ids_by_field(races, full_ids)
    for n in set(races["n_valid_horses"].astype(int)):
        if len(clean_peers.get(int(n), [])) < 2:
            raise ValueError(f"field size {n} lacks two clean donor races")
        if len(full_peers.get(int(n), [])) < 2:
            raise ValueError(f"field size {n} lacks two full-sample donor races")

    panel_a_frames: list[pd.DataFrame] = []
    panel_b_frames: list[pd.DataFrame] = []
    for target_name in TARGET_MARKETS:
        target = load_market(args.data_root, target_name, all_ids)
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
    summary_b = None if bounds_b is None else summarize_panel_b(bounds_b)
    improve_a = benchmark_improvements_a(metrics_a)
    improve_b = None if bounds_b is None else benchmark_improvements_b(bounds_b)
    p3 = order_information_test(metrics_a)
    thresholds = absolute_threshold_decision(summary_a, summary_b)
    selection = sample_selection_summary(races, set(clean_ids), set(full_ids))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    outputs: list[tuple[Path, pd.DataFrame]] = [
        (args.output_dir / "main_metrics.csv", metrics_a),
        (args.output_dir / "main_panel_a_summary.csv", summary_a),
        (args.output_dir / "main_panel_a_improvements.csv", improve_a),
        (args.output_dir / "main_order_information.csv", p3),
        (args.output_dir / "main_threshold_decisions.csv", thresholds),
        (args.output_dir / "main_sample_selection.csv", selection),
    ]
    if bounds_b is not None and summary_b is not None and improve_b is not None:
        outputs.extend(
            [
                (args.output_dir / "main_metrics_bounds.csv", bounds_b),
                (args.output_dir / "main_panel_b_summary.csv", summary_b),
                (args.output_dir / "main_panel_b_improvements.csv", improve_b),
            ]
        )
    for path, frame in outputs:
        frame.to_csv(path, index=False, float_format="%.12g")
        paths.append(path)

    write_latex_tables(
        args.table_dir, summary_a, summary_b, improve_a, improve_b, p3
    )
    for name in (
        "main_panel_a.tex",
        "main_benchmark_comparison.tex",
        "main_order_information.tex",
    ):
        paths.append(args.table_dir / name)
    if summary_b is not None:
        paths.append(args.table_dir / "main_panel_b.tex")
    write_manifest(
        args.output_dir,
        paths,
        max_races=args.max_races,
        bounds=not args.skip_bounds,
    )

    print(f"Panel A races: {len(clean_ids):,}")
    print(f"Panel B races: {0 if args.skip_bounds else len(full_ids):,}")
    print(summary_a[summary_a["model"].eq("main")].to_string(index=False))
    if summary_b is not None:
        print(summary_b[summary_b["model"].eq("main")].to_string(index=False))
    print("PASS: main cross-pool analysis completed")


if __name__ == "__main__":
    main()
