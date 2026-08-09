"""Regression tests for the frozen uncapped-win Harville invariant."""
from __future__ import annotations

import pandas as pd
import pytest

from analysis import main_analysis_core as core
from analysis import main_analysis_p3_joint as p3_joint
from analysis import main_analysis_p3_joint_fast as p3_joint_fast


def test_p3_modules_share_guarded_harville_primitive() -> None:
    assert getattr(core.harville_trifecta, "_win_uncapped_guarded", False)
    assert p3_joint.harville_trifecta is core.harville_trifecta
    assert p3_joint_fast.harville_trifecta is core.harville_trifecta


def test_guarded_harville_rejects_capped_win_before_calculation() -> None:
    source = pd.DataFrame(columns=["first_no", "second_no", "third_no"])
    win = pd.DataFrame(
        {
            "horse_no": [1, 2, 3],
            "odds": [2.0, 3.0, 4.0],
            "is_capped_odds": [False, True, False],
        }
    )
    with pytest.raises(ValueError, match="uncapped win odds"):
        core.harville_trifecta(source, win)
