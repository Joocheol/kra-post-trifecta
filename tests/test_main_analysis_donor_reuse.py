from __future__ import annotations

import unittest

import pandas as pd

from analysis.main_analysis_donor_reuse import donor_reuse_diagnostic


class DonorReuseDiagnosticTest(unittest.TestCase):
    @staticmethod
    def _frame(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
        return pd.DataFrame(
            rows,
            columns=["race_id", "model", "n_valid_horses", "donor_race_id"],
        )

    def test_summarizes_unique_target_to_donor_mappings(self) -> None:
        panel_a = self._frame(
            [
                ("r1", "other_race", 8, "d1"),
                ("r1", "other_race", 8, "d1"),
                ("r2", "other_race", 8, "d1"),
                ("r3", "other_race", 8, "d2"),
                ("r4", "main", 8, ""),
            ]
        )
        panel_b = self._frame(
            [
                ("r5", "other_race", 10, "d3"),
                ("r6", "other_race", 10, "d4"),
            ]
        )
        result = donor_reuse_diagnostic(panel_a, panel_b)
        a = result[result["panel"].eq("A")].iloc[0]
        b = result[result["panel"].eq("B")].iloc[0]
        self.assertEqual(int(a["n_targets_with_donor"]), 3)
        self.assertEqual(int(a["n_distinct_donors"]), 2)
        self.assertAlmostEqual(float(a["targets_per_distinct_donor"]), 1.5)
        self.assertEqual(int(a["max_targets_per_donor"]), 2)
        self.assertEqual(int(b["n_targets_with_donor"]), 2)
        self.assertEqual(int(b["n_distinct_donors"]), 2)

    def test_rejects_multiple_donors_for_one_target(self) -> None:
        frame = self._frame(
            [
                ("r1", "other_race", 8, "d1"),
                ("r1", "other_race", 8, "d2"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "multiple donors"):
            donor_reuse_diagnostic(frame, None)

    def test_rejects_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing columns"):
            donor_reuse_diagnostic(pd.DataFrame({"race_id": ["r1"]}), None)


if __name__ == "__main__":
    unittest.main()
