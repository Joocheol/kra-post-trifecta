"""Explicit invariants for the frozen main-analysis design."""
from __future__ import annotations

import pandas as pd


def assert_win_uncapped(win_frame: pd.DataFrame) -> None:
    """Fail rather than silently point-value a capped win observation."""
    if "is_capped_odds" not in win_frame.columns:
        raise ValueError("win frame is missing is_capped_odds")
    if bool(win_frame["is_capped_odds"].fillna(False).any()):
        raise ValueError(
            "Harville benchmark requires uncapped win odds; current frozen design "
            "does not silently point-value capped win observations"
        )
