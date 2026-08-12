from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.reviewer_diagnostics import (
    EFFECTIVE_TICKET_GRID,
    TARGET_MARKETS,
    display_cap_summary,
    expected_two_pool_tv,
    finite_pool_reference,
    odds_level_summary,
)


class ReviewerDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def levels() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "race_id": ["e1", "e2", "q1"],
                "sum_inverse_odds": [1.1, 1.3, 0.9],
                "n_combinations": [6, 12, 3],
                "n_capped_rows": [0, 2, 1],
                "any_capped": [False, True, True],
                "market": ["exacta", "exacta", "quinella"],
            }
        )

    def test_expected_two_pool_tv_has_closed_form_for_equal_binary_pool(self) -> None:
        probability = np.array([0.5, 0.5])
        for effective_n in (100, 10_000):
            expected = 1.0 / np.sqrt(np.pi * effective_n)
            self.assertAlmostEqual(
                expected_two_pool_tv(probability, effective_n), expected
            )

    def test_odds_level_summary_keeps_cap_groups_separate(self) -> None:
        result = odds_level_summary(self.levels()).set_index(
            ["market", "any_capped"]
        )
        self.assertEqual(int(result.loc[("exacta", False), "n_races"]), 1)
        self.assertEqual(int(result.loc[("exacta", True), "n_races"]), 1)
        self.assertAlmostEqual(
            float(result.loc[("exacta", False), "median_sum_inverse_odds"]), 1.1
        )
        self.assertAlmostEqual(
            float(result.loc[("exacta", True), "median_capped_rows"]), 2.0
        )

    def test_display_cap_summary_counts_races_rows_and_shares(self) -> None:
        result = display_cap_summary(self.levels()).set_index("market")
        self.assertEqual(int(result.loc["exacta", "n_races"]), 2)
        self.assertEqual(int(result.loc["exacta", "capped_races"]), 1)
        self.assertEqual(int(result.loc["exacta", "capped_rows"]), 2)
        self.assertAlmostEqual(float(result.loc["exacta", "capped_race_share"]), 0.5)
        self.assertAlmostEqual(
            float(result.loc["quinella", "capped_race_share"]), 1.0
        )

    def test_finite_pool_reference_uses_race_normalized_prices(self) -> None:
        observed = pd.DataFrame(
            {
                "model": ["main"] * len(TARGET_MARKETS),
                "target_market": list(TARGET_MARKETS),
                "median_tv": [0.1, 0.2, 0.3],
            }
        )
        market_frame = pd.DataFrame(
            {"race_id": ["r1", "r1"], "odds": [2.0, 2.0]}
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            observed_path = root / "observed.csv"
            observed.to_csv(observed_path, index=False)
            with (
                patch(
                    "analysis.reviewer_diagnostics.clean_ids",
                    return_value={"r1"},
                ),
                patch(
                    "analysis.reviewer_diagnostics.read_parquets",
                    return_value=market_frame,
                ),
            ):
                result = finite_pool_reference(
                    root, root / "sample.csv", observed_path
                )

        self.assertEqual(len(result), len(TARGET_MARKETS) * len(EFFECTIVE_TICKET_GRID))
        self.assertTrue((result["n_races"] == 1).all())
        first_n = EFFECTIVE_TICKET_GRID[0]
        first = result[result["effective_tickets_per_pool"].eq(first_n)]
        expected_reference = 1.0 / np.sqrt(np.pi * first_n)
        np.testing.assert_allclose(first["median_reference_tv"], expected_reference)
        np.testing.assert_allclose(first["p05_reference_tv"], expected_reference)
        np.testing.assert_allclose(first["p95_reference_tv"], expected_reference)
        np.testing.assert_allclose(
            first["observed_to_reference_ratio"],
            first["observed_main_median_tv"] / expected_reference,
        )


if __name__ == "__main__":
    unittest.main()
