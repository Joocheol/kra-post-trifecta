"""Numerical primitives for the co-primary cross-pool main analysis."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from analysis.data_audit import MARKET_SPECS
from analysis.main_analysis_guards import assert_win_uncapped

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
    """Translate displayed total-payout multiples to reciprocal-price intervals.

    The frozen implementation treats uncapped one-decimal displays as rounded to
    the nearest tenth (plus/minus 0.05). A capped 9999.9 observation supplies only
    a lower bound on the true payout multiple and therefore a zero lower bound on
    its reciprocal price.
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
    """Aggregate a partition of source states before normalization.

    Each ordered trifecta state belongs to exactly one outcome in every target
    market used here. Therefore summing raw reciprocal-price intervals within
    each partition cell is an exact representation of A q; no relaxation is
    introduced by reducing the LP to target-market dimension.
    """
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
    lower = price_set.lower
    upper = price_set.upper
    sum_lower = float(lower.sum())
    sum_upper = float(upper.sum())
    min_den = lower + (sum_upper - upper)
    max_den = upper + (sum_lower - lower)
    p_min = np.divide(lower, min_den, out=np.zeros_like(lower), where=min_den > 0)
    p_max = np.divide(upper, max_den, out=np.ones_like(upper), where=max_den > 0)
    return np.clip(p_min, 0.0, 1.0), np.clip(p_max, 0.0, 1.0)


def tv_upper_outer(left: PriceSet, right: PriceSet) -> float:
    """Certified outer TV upper bound from componentwise difference ranges."""
    if left.size != right.size:
        raise ValueError("TV price sets must have the same dimension")
    lmin, lmax = price_set_component_bounds(left)
    rmin, rmax = price_set_component_bounds(right)
    diff_min = lmin - rmax
    diff_max = lmax - rmin
    bound = 0.5 * float(np.maximum(np.abs(diff_min), np.abs(diff_max)).sum())
    return min(1.0, max(0.0, bound))


def _scale_bounds(price_set: PriceSet) -> tuple[float, float | None]:
    sum_upper = float(price_set.upper.sum())
    sum_lower = float(price_set.lower.sum())
    return 1.0 / sum_upper, None if sum_lower <= 0 else 1.0 / sum_lower


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
    a_eq = sparse.coo_matrix(
        (np.ones(2 * cdim), (eq_rows, eq_cols)), shape=(2, nvar)
    ).tocsr()
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
    js = 0.5 * float(
        np.sum(actual * np.log(actual / midpoint))
        + np.sum(predicted * np.log(predicted / midpoint))
    )
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
    corr = (
        float("nan")
        if len(actual) < 2 or np.std(actual) <= 0 or np.std(predicted) <= 0
        else float(np.corrcoef(actual, predicted)[0, 1])
    )
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
        return [int(value) for value in frame[spec.keys[0]].to_numpy()]
    return [
        tuple(int(value) for value in values)
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
    assert_win_uncapped(win)
    win_keys = [int(value) for value in win["horse_no"].to_numpy()]
    probabilities = normalize_inverse_odds(win["odds"].to_numpy(dtype=float))
    pmap = dict(zip(win_keys, probabilities))
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
    value = np.bincount(groups, weights=np.asarray(point, dtype=float), minlength=n_groups)
    return value / value.sum()


def choose_other_race(race_id: str, peers: list[str], label: str) -> str | None:
    """Choose a deterministic same-field donor; return None if none exists."""
    alternatives = [peer for peer in peers if peer != race_id]
    if not alternatives:
        return None
    return alternatives[stable_uint(f"donor|{label}|{race_id}") % len(alternatives)]


def deterministic_permutation(length: int, race_id: str, target: str, panel: str) -> np.ndarray:
    rng = np.random.default_rng(stable_uint(f"perm|{panel}|{target}|{race_id}"))
    return rng.permutation(length)
