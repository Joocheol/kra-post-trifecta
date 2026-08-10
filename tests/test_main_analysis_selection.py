from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.main_analysis_selection import (
    _period_share,
    _share,
    heterogeneity_summary,
    render_table,
)


class SelectionDiagnosticTest(unittest.TestCase):
    @staticmethod
    def _composition() -> pd.DataFrame:
        rows = []
        for group in ("clean", "capped"):
            rows.extend(
                [
                    (group, "year", "2018", 0.2),
                    (group, "year", "2019", 0.3),
                    (group, "year", "2022", 0.5),
                    (group, "meet", "1", 0.4),
                    (group, "meet", "2", 0.35),
                    (group, "meet", "3", 0.25),
                ]
            )
        return pd.DataFrame(
            rows, columns=["sample_group", "dimension", "level", "share"]
        )

    def test_share_and_period_share_use_unique_cells(self) -> None:
        composition = self._composition()
        self.assertAlmostEqual(_share(composition, "clean", "meet", "2"), 0.35)
        self.assertAlmostEqual(
            _period_share(composition, "clean", {2018, 2019}), 0.5
        )
        duplicate = pd.concat([composition, composition.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "missing unique composition cell"):
            _share(duplicate, "clean", "year", "2018")

    def test_render_table_contains_frozen_summary_values(self) -> None:
        selection = pd.DataFrame(
            {
                "sample_group": ["clean", "capped"],
                "n_races": [10, 20],
                "field_size_median": [8, 10],
                "field_size_q25": [7, 9],
                "field_size_q75": [9, 11],
            }
        )
        tails = pd.DataFrame(
            {
                "sample_group": ["clean", "capped"],
                "odds_q95": [123.4, 567.8],
                "odds_q99": [234.5, 678.9],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "selection.tex"
            render_table(selection, self._composition(), tails, output)
            rendered = output.read_text(encoding="utf-8")
        self.assertIn(r"clean & 10 & 8 [7, 9] & 50.0\%", rendered)
        self.assertIn(r"상한 포함 & 20 & 10 [9, 11] & 50.0\%", rendered)
        self.assertIn("123.4 & 234.5", rendered)

    def test_heterogeneity_summary_orders_and_ranges_main_rows(self) -> None:
        rows = []
        for target in ("trio", "exacta", "quinella", "win"):
            for dimension in ("meet", "year", "n_valid_horses"):
                rows.extend(
                    [
                        (target, "main", dimension, 0.04),
                        (target, "main", dimension, 0.07),
                        (target, "uniform", dimension, 0.50),
                    ]
                )
        frame = pd.DataFrame(
            rows, columns=["target_market", "model", "dimension", "median_tv"]
        )
        result = heterogeneity_summary(frame)
        first = result.iloc[0]
        self.assertEqual((first["target_market"], first["dimension"]), ("exacta", "year"))
        self.assertEqual(int(first["n_levels"]), 2)
        self.assertAlmostEqual(float(first["min_group_median_tv"]), 0.04)
        self.assertAlmostEqual(float(first["max_group_median_tv"]), 0.07)


if __name__ == "__main__":
    unittest.main()
