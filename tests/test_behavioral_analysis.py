from __future__ import annotations

import itertools
import unittest

import numpy as np
import pandas as pd

from analysis.behavioral_analysis import (
    FoldModel,
    attach_event_probabilities,
    attach_unordered_event_probabilities,
    cluster_bootstrap_median_interval,
    price_summary,
    race_metric_rows,
    score_unordered_price_model,
    training_year_mask,
)
from analysis.behavioral_core import (
    MonotoneCalibrator,
    PowerWeight,
    PrelecWeight,
    StageTemperature,
)


class BehavioralAnalysisTests(unittest.TestCase):
    def test_training_year_masks_exclude_validation_and_future_years(self) -> None:
        years = pd.Series([2016, 2017, 2018, 2019])
        self.assertEqual(
            training_year_mask(years, 2018, "leave_one_year_out").tolist(),
            [True, True, False, True],
        )
        self.assertEqual(
            training_year_mask(years, 2018, "time_forward").tolist(),
            [True, True, False, False],
        )

    @staticmethod
    def fold(stage2: StageTemperature, stage3: StageTemperature) -> FoldModel:
        identity = MonotoneCalibrator(
            np.array([0.001, 0.999]), np.array([0.001, 0.999])
        )
        return FoldModel(
            validation_year=2025,
            calibration=identity,
            stage2=stage2,
            stage3=stage3,
            win_weight=identity,
            prelec=PrelecWeight(1.0, 1.0),
            power=PowerWeight(1.0),
        )

    @staticmethod
    def horses() -> pd.DataFrame:
        probability = np.array([0.35, 0.25, 0.18, 0.13, 0.09])
        return pd.DataFrame(
            {
                "race_id": ["r1"] * 5,
                "horse_no": np.arange(1, 6),
                "objective_probability": probability,
                "n_valid_horses": [5] * 5,
            }
        )

    def test_harville_exacta_and_trifecta_probabilities_sum_to_one(self) -> None:
        fold = self.fold(StageTemperature(0.0, 0.0), StageTemperature(0.0, 0.0))
        exacta = pd.DataFrame(
            [("r1", first, second) for first, second in itertools.permutations(range(1, 6), 2)],
            columns=["race_id", "first_no", "second_no"],
        )
        exacta_result = attach_event_probabilities(
            exacta, self.horses(), "harville", fold
        )
        self.assertAlmostEqual(float(exacta_result["p_joint"].sum()), 1.0)

        trifecta = pd.DataFrame(
            [
                ("r1", first, second, third)
                for first, second, third in itertools.permutations(range(1, 6), 3)
            ],
            columns=["race_id", "first_no", "second_no", "third_no"],
        )
        trifecta_result = attach_event_probabilities(
            trifecta, self.horses(), "harville", fold
        )
        self.assertAlmostEqual(float(trifecta_result["p_joint"].sum()), 1.0)

    def test_stage_adjusted_probabilities_also_sum_to_one(self) -> None:
        fold = self.fold(StageTemperature(-0.3, 0.1), StageTemperature(-0.5, -0.1))
        trifecta = pd.DataFrame(
            [
                ("r1", first, second, third)
                for first, second, third in itertools.permutations(range(1, 6), 3)
            ],
            columns=["race_id", "first_no", "second_no", "third_no"],
        )
        result = attach_event_probabilities(
            trifecta, self.horses(), "stage_temperature", fold
        )
        self.assertAlmostEqual(float(result["p_joint"].sum()), 1.0)
        self.assertTrue(bool(result["p2_cond"].between(0.0, 1.0).all()))
        self.assertTrue(bool(result["p3_cond"].between(0.0, 1.0).all()))

    def test_unordered_probabilities_equal_sums_over_ordered_claims(self) -> None:
        fold = self.fold(StageTemperature(-0.3, 0.1), StageTemperature(-0.5, -0.1))
        quinella = pd.DataFrame(
            [
                ("r1", first, second)
                for first, second in itertools.combinations(range(1, 6), 2)
            ],
            columns=["race_id", "horse_a", "horse_b"],
        )
        unordered = attach_unordered_event_probabilities(
            quinella, self.horses(), "stage_temperature", fold, "quinella"
        )
        exacta = pd.DataFrame(
            [
                ("r1", first, second)
                for first, second in itertools.permutations(range(1, 6), 2)
            ],
            columns=["race_id", "first_no", "second_no"],
        )
        ordered = attach_event_probabilities(
            exacta, self.horses(), "stage_temperature", fold
        )
        ordered["horse_a"] = ordered[["first_no", "second_no"]].min(axis=1)
        ordered["horse_b"] = ordered[["first_no", "second_no"]].max(axis=1)
        expected = ordered.groupby(["horse_a", "horse_b"])["p_joint"].sum()
        actual = unordered.set_index(["horse_a", "horse_b"])["p_joint"]
        np.testing.assert_allclose(actual.loc[expected.index], expected)
        self.assertAlmostEqual(float(actual.sum()), 1.0)

    def test_unordered_identity_weight_matches_probability_aggregation(self) -> None:
        fold = self.fold(StageTemperature(-0.3, 0.1), StageTemperature(-0.5, -0.1))
        trio = pd.DataFrame(
            [
                ("r1", first, second, third)
                for first, second, third in itertools.combinations(range(1, 6), 3)
            ],
            columns=["race_id", "horse_a", "horse_b", "horse_c"],
        )
        events = attach_unordered_event_probabilities(
            trio, self.horses(), "stage_temperature", fold, "trio"
        )
        reduced, _ = score_unordered_price_model(
            events, "trio", "M-R", lambda values: values
        )
        sequential2, _ = score_unordered_price_model(
            events, "trio", "M-S2", lambda values: values
        )
        sequential3, _ = score_unordered_price_model(
            events, "trio", "M-S3", lambda values: values
        )
        np.testing.assert_allclose(reduced, sequential2, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(reduced, sequential3, rtol=1e-13, atol=0.0)

    def test_production_metric_path_and_pool_level_invariance(self) -> None:
        fold = self.fold(StageTemperature(0.0, 0.0), StageTemperature(0.0, 0.0))
        frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r2", "r2"],
                "race_date": ["2025-01-01"] * 2 + ["2025-01-02"] * 2,
                "actual_price_share": [0.6, 0.4, 0.25, 0.75],
                "raw_price": [0.6, 0.4, 0.25, 0.75],
            }
        )
        score = np.array([0.7, 0.3, 0.5, 0.5])
        arguments = [np.array([0.2, 0.3, 0.4, 0.5])]

        low_level = race_metric_rows(
            frame,
            score,
            arguments,
            fold,
            "stage_temperature",
            "prelec",
            "M-R",
            "exacta",
            1.0,
        )
        high_level = race_metric_rows(
            frame,
            score,
            arguments,
            fold,
            "stage_temperature",
            "prelec",
            "M-R",
            "exacta",
            2.0,
        )

        normalized_metrics = ["tv", "mae", "log_rmse", "js", "support_share"]
        np.testing.assert_allclose(
            low_level[normalized_metrics], high_level[normalized_metrics]
        )
        self.assertFalse(
            np.allclose(low_level["raw_mae"], high_level["raw_mae"])
        )

        first_race = low_level.set_index("race_id").loc["r1"]
        actual = np.array([0.6, 0.4])
        predicted = np.array([0.7, 0.3])
        midpoint = 0.5 * (actual + predicted)
        self.assertAlmostEqual(first_race["tv"], 0.1)
        self.assertAlmostEqual(first_race["mae"], 0.1)
        self.assertAlmostEqual(
            first_race["log_rmse"],
            float(np.sqrt(np.mean(np.square(np.log(predicted) - np.log(actual))))),
        )
        self.assertAlmostEqual(
            first_race["js"],
            0.5
            * float(
                np.sum(actual * np.log(actual / midpoint))
                + np.sum(predicted * np.log(predicted / midpoint))
            ),
        )

        summary = price_summary(low_level).iloc[0]
        self.assertEqual(int(summary["n_races"]), 2)
        self.assertAlmostEqual(
            float(summary["median_tv"]), float(low_level["tv"].median())
        )
        self.assertAlmostEqual(
            float(summary["median_raw_mae"]),
            float(low_level["raw_mae"].median()),
        )

    def test_production_price_summary_uses_race_equal_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "probability_model": ["stage_temperature"] * 2,
                "tail_model": ["prelec"] * 2,
                "price_model": ["M-R"] * 2,
                "target_market": ["exacta"] * 2,
                "validation_year": [2024, 2025],
                "race_id": ["r1", "r2"],
                "tv": [0.1, 0.3],
                "mae": [0.01, 0.03],
                "log_rmse": [0.2, 0.4],
                "js": [0.02, 0.06],
                "raw_mae": [0.04, 0.08],
                "support_share": [1.0, 0.5],
            }
        )
        result = price_summary(frame).iloc[0]
        self.assertEqual(int(result["n_races"]), 2)
        self.assertEqual(int(result["n_years"]), 2)
        self.assertAlmostEqual(float(result["median_tv"]), 0.2)
        self.assertAlmostEqual(float(result["median_raw_mae"]), 0.06)
        self.assertAlmostEqual(float(result["mean_support_share"]), 0.75)

    def test_cluster_bootstrap_is_deterministic_and_respects_constant_effect(self) -> None:
        values = np.full(12, 0.125)
        clusters = np.repeat(["2025-01-01", "2025-01-02", "2025-01-03"], 4)
        first = cluster_bootstrap_median_interval(values, clusters, "constant", reps=99)
        second = cluster_bootstrap_median_interval(values, clusters, "constant", reps=99)
        self.assertEqual(first, second)
        self.assertEqual(first, (0.125, 0.125))


if __name__ == "__main__":
    unittest.main()
