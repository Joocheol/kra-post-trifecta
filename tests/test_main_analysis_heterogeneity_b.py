from __future__ import annotations

import unittest

import pandas as pd

from analysis.main_analysis_heterogeneity_b import comparison_summary, panel_b_heterogeneity


class PanelBHeterogeneityTest(unittest.TestCase):
    def test_panel_b_groups_year_meet_and_field_size(self) -> None:
        bounds = pd.DataFrame(
            {
                "race_id": ["2019-01-01_1_01", "2019-01-02_2_01", "2022-01-01_1_01", "2022-01-02_2_01"],
                "target_market": ["exacta"] * 4,
                "model": ["main"] * 4,
                "n_valid_horses": [8, 8, 10, 10],
                "tv_lower": [0.04, 0.06, 0.05, 0.07],
                "tv_upper_outer": [0.08, 0.10, 0.09, 0.11],
            }
        )
        result = panel_b_heterogeneity(bounds)
        year = result[(result["dimension"] == "year") & (result["level"] == "2019")].iloc[0]
        self.assertAlmostEqual(year["median_tv_lower"], 0.05)
        self.assertAlmostEqual(year["median_tv_upper_outer"], 0.09)
        self.assertEqual(int(year["n_races"]), 2)

    def test_comparison_summary_uses_main_ranges(self) -> None:
        rows_a = []
        rows_b = []
        for target in ("exacta", "quinella", "trio", "win"):
            for dimension in ("year", "meet", "n_valid_horses"):
                for level, value in (("1", 0.04), ("2", 0.06)):
                    rows_a.append({"target_market": target, "model": "main", "dimension": dimension, "level": level, "median_tv": value})
                    rows_b.append({"target_market": target, "model": "main", "dimension": dimension, "level": level, "median_tv_lower": value - 0.01, "median_tv_upper_outer": value + 0.02})
        result = comparison_summary(pd.DataFrame(rows_a), pd.DataFrame(rows_b))
        row = result[(result["target_market"] == "trio") & (result["dimension"] == "meet")].iloc[0]
        self.assertAlmostEqual(row["panel_a_median_tv_min"], 0.04)
        self.assertAlmostEqual(row["panel_a_median_tv_max"], 0.06)
        self.assertAlmostEqual(row["panel_b_median_tv_lower_min"], 0.03)
        self.assertAlmostEqual(row["panel_b_median_tv_upper_max"], 0.08)


if __name__ == "__main__":
    unittest.main()
