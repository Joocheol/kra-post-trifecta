"""Regression tests for the frozen uncapped-win Harville invariant."""
from __future__ import annotations

import unittest

import pandas as pd

from analysis import main_analysis_core as core
from analysis import main_analysis_p3_joint as p3_joint
from analysis import main_analysis_p3_joint_fast as p3_joint_fast


class HarvilleGuardRegressionTest(unittest.TestCase):
    def test_p3_modules_share_guarded_harville_primitive(self) -> None:
        self.assertIs(p3_joint.harville_trifecta, core.harville_trifecta)
        self.assertIs(p3_joint_fast.harville_trifecta, core.harville_trifecta)

    def test_guarded_harville_rejects_capped_win_before_calculation(self) -> None:
        source = pd.DataFrame(columns=["first_no", "second_no", "third_no"])
        win = pd.DataFrame(
            {
                "horse_no": [1, 2, 3],
                "odds": [2.0, 3.0, 4.0],
                "is_capped_odds": [False, True, False],
            }
        )
        with self.assertRaisesRegex(ValueError, "uncapped win odds"):
            core.harville_trifecta(source, win)


if __name__ == "__main__":
    unittest.main()
