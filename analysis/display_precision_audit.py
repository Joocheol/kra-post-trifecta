"""Verify the one-decimal display grid used by the Panel B rounding envelope."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data_audit import ANALYSIS_MARKETS, read_parquets
from analysis.main_analysis_core import DISPLAY_CAP

DISPLAY_STEP = 0.1
GRID_TOL = 1e-8
BANDS = (
    (0.0, 10.0, "(0,10)"),
    (10.0, 100.0, "[10,100)"),
    (100.0, 1000.0, "[100,1000)"),
    (1000.0, 5000.0, "[1000,5000)"),
    (5000.0, DISPLAY_CAP + DISPLAY_STEP, "[5000,9999.9]"),
)


def off_tenth_grid_mask(odds: np.ndarray, capped: np.ndarray) -> np.ndarray:
    """Return uncapped, finite, positive displays that are not on a 0.1 grid."""
    odds = np.asarray(odds, dtype=float)
    capped = np.asarray(capped, dtype=bool)
    eligible = (~capped) & np.isfinite(odds) & (odds > 0)
    scaled = odds * 10.0
    on_grid = np.isclose(scaled, np.rint(scaled), rtol=0.0, atol=GRID_TOL)
    return eligible & ~on_grid


def audit_display_precision(data_root: Path) -> pd.DataFrame:
    """Audit the 0.1 display grid by market and payout-multiple magnitude band."""
    rows: list[dict[str, object]] = []
    for market in ANALYSIS_MARKETS:
        frame = read_parquets(data_root, market, columns=["odds", "is_capped_odds"])
        odds = pd.to_numeric(frame["odds"], errors="coerce").to_numpy(dtype=float)
        capped = frame["is_capped_odds"].fillna(False).to_numpy(dtype=bool)
        off_grid = off_tenth_grid_mask(odds, capped)
        eligible = (~capped) & np.isfinite(odds) & (odds > 0)
        for lower, upper, label in BANDS:
            in_band = eligible & (odds >= lower) & (odds < upper)
            # Include the displayed cap endpoint in the final magnitude band, while
            # capped rows remain excluded from the rounding-grid audit itself.
            if upper > DISPLAY_CAP:
                in_band = eligible & (odds >= lower) & (odds <= DISPLAY_CAP)
            rows.append(
                {
                    "market": market,
                    "odds_band": label,
                    "n_uncapped_rows": int(in_band.sum()),
                    "n_off_tenth_grid": int((in_band & off_grid).sum()),
                }
            )
    return pd.DataFrame(rows)


def assert_display_precision(audit: pd.DataFrame) -> None:
    failures = audit[audit["n_off_tenth_grid"].gt(0)]
    if not failures.empty:
        details = ", ".join(
            f"{row.market}/{row.odds_band}={int(row.n_off_tenth_grid)}"
            for row in failures.itertuples(index=False)
        )
        raise ValueError(f"uncapped KRA odds violate the 0.1 display grid: {details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("KRA/parsed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_display_precision(args.data_root)
    assert_display_precision(audit)
    print(audit.to_string(index=False))
    print("display_precision_audit=pass")


if __name__ == "__main__":
    main()
