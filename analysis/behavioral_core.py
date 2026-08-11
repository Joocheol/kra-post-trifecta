"""Numerical primitives for the cross-fitted behavioral-model analysis.

The behavioral analysis deliberately separates two objects:

1. objective rank probabilities estimated from realized finish orders; and
2. a monotone map from those probabilities to win-pool price shares.

The first object is estimated outside the validation year.  The second object is
estimated only on the training years and then transferred without a nonlinear
refit to exacta and trifecta prices.  ``M-U`` and the reduced probability-weighting
model ``M-R`` are observationally equivalent when both maps are recovered
nonparametrically from the same win prices; the directly identified comparison is
therefore reduced versus sequential evaluation (``M-S2``/``M-S3``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression, minimize
from scipy.special import logsumexp


EPSILON = 1e-12
TINY = np.finfo(float).tiny
REFERENCE_FIELD_SIZE = 10.0


def _as_finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def normalize_scores(values: np.ndarray) -> np.ndarray:
    """Normalize a positive score vector and fail loudly on invalid input."""
    values = _as_finite_vector(values, "scores")
    if np.any(values <= 0):
        raise ValueError("scores must be strictly positive")
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("score sum must be finite and positive")
    return values / total


def normalized_inverse_odds(odds: np.ndarray) -> np.ndarray:
    """Return the within-pool price share implied by gross payout multiples."""
    odds = _as_finite_vector(odds, "odds")
    if np.any(odds <= 0):
        raise ValueError("odds must be strictly positive")
    return normalize_scores(1.0 / odds)


@dataclass(frozen=True)
class MonotoneCalibrator:
    """Piecewise-linear monotone regression with explicit support tracking."""

    x: np.ndarray
    y: np.ndarray

    @classmethod
    def fit(
        cls,
        x: Iterable[float],
        y: Iterable[float],
        sample_weight: Iterable[float] | None = None,
    ) -> "MonotoneCalibrator":
        x_array = _as_finite_vector(np.asarray(x, dtype=float), "x")
        y_array = _as_finite_vector(np.asarray(y, dtype=float), "y")
        if len(x_array) != len(y_array):
            raise ValueError("x and y lengths differ")
        if sample_weight is None:
            weight_array = np.ones(len(x_array), dtype=float)
        else:
            weight_array = _as_finite_vector(
                np.asarray(sample_weight, dtype=float), "sample_weight"
            )
            if len(weight_array) != len(x_array):
                raise ValueError("sample_weight length differs from x")
            if np.any(weight_array <= 0):
                raise ValueError("sample weights must be strictly positive")

        order = np.argsort(x_array, kind="mergesort")
        xs = x_array[order]
        ys = y_array[order]
        ws = weight_array[order]
        unique_x, inverse = np.unique(xs, return_inverse=True)
        sum_weight = np.bincount(inverse, weights=ws)
        mean_y = np.bincount(inverse, weights=ws * ys) / sum_weight
        fitted = isotonic_regression(mean_y, weights=sum_weight, increasing=True).x
        fitted = np.clip(np.asarray(fitted, dtype=float), EPSILON, None)
        return cls(unique_x.astype(float), fitted)

    @property
    def support(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[-1])

    def predict(self, x: Iterable[float], *, clip: bool = True) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("prediction input contains non-finite values")
        if not clip and (np.any(values < self.x[0]) or np.any(values > self.x[-1])):
            raise ValueError("prediction input lies outside monotone-fit support")
        result = np.interp(values, self.x, self.y, left=self.y[0], right=self.y[-1])
        return np.clip(result, EPSILON, None)


@dataclass(frozen=True)
class StageTemperature:
    """Field-size-dependent Plackett--Luce temperature for one rank stage."""

    intercept: float
    field_slope: float

    def alpha(self, field_size: int | np.ndarray) -> np.ndarray:
        n = np.asarray(field_size, dtype=float)
        if np.any(n < 3):
            raise ValueError("field size must be at least three")
        return np.exp(self.intercept + self.field_slope * np.log(n / REFERENCE_FIELD_SIZE))


def _choice_nll(
    parameters: np.ndarray,
    batches: dict[int, tuple[np.ndarray, np.ndarray]],
) -> float:
    model = StageTemperature(float(parameters[0]), float(parameters[1]))
    total = 0.0
    observations = 0
    for choice_size, (strengths, chosen_index) in batches.items():
        alpha = float(model.alpha(choice_size))
        log_strength = np.log(np.clip(strengths, EPSILON, None))
        row_index = np.arange(len(strengths))
        total += float(
            np.sum(
                logsumexp(alpha * log_strength, axis=1)
                - alpha * log_strength[row_index, chosen_index]
            )
        )
        observations += len(strengths)
    return float(total / max(observations, 1))


def fit_stage_temperature(
    choice_sets: list[tuple[np.ndarray, int]],
) -> StageTemperature:
    """Fit a deterministic two-parameter conditional-choice temperature."""
    if not choice_sets:
        raise ValueError("no choice sets supplied")
    grouped: dict[int, list[tuple[np.ndarray, int]]] = {}
    for strengths, chosen_index in choice_sets:
        values = normalize_scores(np.asarray(strengths, dtype=float))
        if chosen_index < 0 or chosen_index >= len(values):
            raise ValueError("chosen index is outside its choice set")
        grouped.setdefault(len(values), []).append((values, chosen_index))
    batches = {
        size: (
            np.stack([values for values, _ in rows]),
            np.asarray([chosen for _, chosen in rows], dtype=np.int64),
        )
        for size, rows in grouped.items()
    }
    result = minimize(
        _choice_nll,
        x0=np.zeros(2, dtype=float),
        args=(batches,),
        method="L-BFGS-B",
        bounds=((-2.5, 2.5), (-1.5, 1.5)),
        options={"ftol": 1e-10, "gtol": 1e-6, "maxiter": 300},
    )
    if not result.success or np.any(~np.isfinite(result.x)):
        result = minimize(
            _choice_nll,
            x0=np.asarray(result.x if np.all(np.isfinite(result.x)) else np.zeros(2)),
            args=(batches,),
            method="Powell",
            bounds=((-2.5, 2.5), (-1.5, 1.5)),
            options={"xtol": 1e-8, "ftol": 1e-10, "maxiter": 500},
        )
    if not result.success or np.any(~np.isfinite(result.x)):
        raise RuntimeError(f"stage-temperature optimization failed: {result.message}")
    return StageTemperature(float(result.x[0]), float(result.x[1]))


def conditional_stage_probability(
    chosen_strength: np.ndarray,
    total_strength: np.ndarray,
    excluded_strengths: Iterable[np.ndarray],
) -> np.ndarray:
    """Probability of one choice after removing previously selected horses."""
    chosen = np.asarray(chosen_strength, dtype=float)
    denominator = np.asarray(total_strength, dtype=float).copy()
    for excluded in excluded_strengths:
        denominator -= np.asarray(excluded, dtype=float)
    if np.any(chosen <= 0) or np.any(denominator <= 0):
        raise ValueError("conditional probability has a non-positive score or denominator")
    result = chosen / denominator
    if np.any(result <= 0) or np.any(result > 1 + 1e-10):
        raise ValueError("invalid conditional probability")
    return np.clip(result, EPSILON, 1.0)


@dataclass(frozen=True)
class PrelecWeight:
    alpha: float
    beta: float

    @classmethod
    def fit(
        cls,
        probability: Iterable[float],
        price_share: Iterable[float],
        sample_weight: Iterable[float] | None = None,
    ) -> "PrelecWeight":
        p = np.clip(_as_finite_vector(np.asarray(probability, dtype=float), "probability"), EPSILON, 1 - EPSILON)
        q = np.clip(_as_finite_vector(np.asarray(price_share, dtype=float), "price_share"), EPSILON, None)
        if len(p) != len(q):
            raise ValueError("probability and price-share lengths differ")
        weights = np.ones(len(p), dtype=float) if sample_weight is None else _as_finite_vector(np.asarray(sample_weight, dtype=float), "sample_weight")
        weights = weights / weights.sum()

        def objective(theta: np.ndarray) -> float:
            alpha, beta, intercept = np.exp(theta[0]), np.exp(theta[1]), theta[2]
            fitted = intercept - beta * np.power(-np.log(p), alpha)
            return float(np.sum(weights * np.square(np.log(q) - fitted)))

        result = minimize(
            objective,
            x0=np.array([0.0, 0.0, 0.0]),
            method="L-BFGS-B",
            bounds=((-2.5, 2.5), (-2.5, 2.5), (-10.0, 10.0)),
            options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 500},
        )
        if not result.success or np.any(~np.isfinite(result.x)):
            raise RuntimeError(f"Prelec optimization failed: {result.message}")
        return cls(float(np.exp(result.x[0])), float(np.exp(result.x[1])))

    def predict(self, probability: Iterable[float]) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), TINY, 1 - EPSILON)
        return np.exp(-self.beta * np.power(-np.log(p), self.alpha))


@dataclass(frozen=True)
class PowerWeight:
    exponent: float

    @classmethod
    def fit(
        cls,
        probability: Iterable[float],
        price_share: Iterable[float],
        sample_weight: Iterable[float] | None = None,
    ) -> "PowerWeight":
        p = np.clip(_as_finite_vector(np.asarray(probability, dtype=float), "probability"), EPSILON, 1.0)
        q = np.clip(_as_finite_vector(np.asarray(price_share, dtype=float), "price_share"), EPSILON, None)
        if len(p) != len(q):
            raise ValueError("probability and price-share lengths differ")
        weights = np.ones(len(p), dtype=float) if sample_weight is None else _as_finite_vector(np.asarray(sample_weight, dtype=float), "sample_weight")
        x = np.column_stack([np.ones(len(p)), np.log(p)])
        root_weight = np.sqrt(weights / weights.mean())
        coefficients, *_ = np.linalg.lstsq(x * root_weight[:, None], np.log(q) * root_weight, rcond=None)
        exponent = float(coefficients[1])
        if not np.isfinite(exponent) or exponent <= 0:
            raise ValueError("estimated power exponent must be finite and positive")
        # The fitted intercept is a nuisance level absorbed by the market-level
        # constant.  Dropping it here preserves the negative-control identity
        # w(ab) = w(a)w(b) exactly for ordered-event decompositions.
        return cls(exponent)

    def predict(self, probability: Iterable[float]) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), TINY, 1.0)
        return np.power(p, self.exponent)


def expected_calibration_error(
    probability: np.ndarray,
    outcome: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Equal-frequency expected calibration error for conditional choices."""
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if len(probability) != len(outcome) or len(probability) == 0:
        raise ValueError("calibration vectors must have the same positive length")
    order = np.argsort(probability, kind="mergesort")
    groups = np.array_split(order, min(bins, len(order)))
    total = float(len(order))
    return float(
        sum(
            len(group)
            / total
            * abs(float(probability[group].mean()) - float(outcome[group].mean()))
            for group in groups
            if len(group)
        )
    )
