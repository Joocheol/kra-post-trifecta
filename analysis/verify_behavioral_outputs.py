#!/usr/bin/env python3
"""Verify frozen behavioral CSVs across numerically equivalent environments."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


CSV_PATHS = (
    "outputs/rank_probability_validation.csv",
    "outputs/rank_probability_validation_by_year.csv",
    "outputs/behavioral_model_parameters.csv",
    "outputs/behavioral_model_comparison.csv",
    "outputs/behavioral_model_improvements.csv",
    "outputs/rank_probability_time_forward.csv",
    "outputs/rank_probability_time_forward_by_year.csv",
    "outputs/behavioral_time_forward_parameters.csv",
    "outputs/behavioral_time_forward_comparison.csv",
    "outputs/behavioral_time_forward_improvements.csv",
)

# All manuscript results are displayed to four decimals or fewer.  This bound
# is stricter than half a unit at the fourth decimal for values on the scale of
# the reported loss metrics, while accommodating optimizer/BLAS differences.
RTOL = 2e-4
ATOL = 2e-6


def compare_frames(expected: pd.DataFrame, actual: pd.DataFrame, path: str) -> None:
    if list(expected.columns) != list(actual.columns):
        raise AssertionError(f"{path}: column mismatch")
    if len(expected) != len(actual):
        raise AssertionError(
            f"{path}: row-count mismatch ({len(expected)} != {len(actual)})"
        )
    for column in expected.columns:
        left = expected[column]
        right = actual[column]
        if pd.api.types.is_integer_dtype(left.dtype):
            if not left.equals(right):
                raise AssertionError(f"{path}: integer column changed: {column}")
        elif pd.api.types.is_numeric_dtype(left.dtype):
            np.testing.assert_allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=RTOL,
                atol=ATOL,
                equal_nan=True,
                err_msg=f"{path}: numeric column changed beyond tolerance: {column}",
            )
        elif not left.equals(right):
            raise AssertionError(f"{path}: categorical column changed: {column}")


def tracked_csv(path: str) -> pd.DataFrame:
    payload = subprocess.check_output(["git", "show", f"HEAD:{path}"])
    return pd.read_csv(io.BytesIO(payload))


def main() -> None:
    for path in CSV_PATHS:
        compare_frames(tracked_csv(path), pd.read_csv(Path(path)), path)
    print(
        f"PASS: {len(CSV_PATHS)} behavioral CSVs match structure exactly "
        f"and floats within rtol={RTOL:g}, atol={ATOL:g}"
    )


if __name__ == "__main__":
    main()
