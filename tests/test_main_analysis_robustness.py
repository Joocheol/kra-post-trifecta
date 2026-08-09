from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analysis.main_analysis_robustness import calibration_regression


class CalibrationRobustnessTest(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = []
        for date, race_id, field in [
            ("2024-01-01", "r1", 5),
            ("2024-01-01", "r2", 5),
            ("2024-01-02", "r3", 6),
        ]:
            predicted = np.array([0.1, 0.2, 0.3, 0.4])
            for value in predicted:
                rows.append(
                    {
                        "race_id": race_id,
                        "race_date": date,
                        "n_valid_horses": field,
                        "n_outcomes": len(predicted),
                        "log_predicted": np.log(value),
                        "log_actual": np.log(value),
                    }
                )
        return pd.DataFrame(rows)

    def test_perfect_calibration_has_unit_slope(self) -> None:
        frame = self._frame()
        for weighting in ("race_equal", "combination_equal"):
            result = calibration_regression(frame, weighting=weighting)
            self.assertAlmostEqual(float(result["beta"]), 1.0, places=10)
            self.assertAlmostEqual(float(result["calibration_r2"]), 1.0, places=10)

    def test_race_date_two_way_is_reported_with_nested_clusters(self) -> None:
        frame = self._frame().copy()
        frame.loc[frame["race_id"].eq("r3"), "log_actual"] += 0.02
        result = calibration_regression(frame, weighting="race_equal")
        self.assertTrue(np.isfinite(float(result["beta_se_race_cluster"])))
        self.assertTrue(np.isfinite(float(result["beta_se_race_date_two_way"])))
        self.assertGreaterEqual(float(result["beta_se_race_cluster"]), 0.0)
        self.assertGreaterEqual(float(result["beta_se_race_date_two_way"]), 0.0)

    def test_rank_deficient_design_is_rejected(self) -> None:
        frame = self._frame().copy()
        frame["log_predicted"] = -2.0
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            calibration_regression(frame, weighting="race_equal")

    def test_unknown_weighting_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown calibration weighting"):
            calibration_regression(self._frame(), weighting="bad")


if __name__ == "__main__":
    unittest.main()
