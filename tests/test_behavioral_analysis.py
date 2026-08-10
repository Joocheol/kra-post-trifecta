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
    score_unordered_price_model,
)
from analysis.behavioral_core import (
    MonotoneCalibrator,
    PowerWeight,
    PrelecWeight,
    StageTemperature,
)


class BehavioralAnalysisTests(unittest.TestCase):
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

    def test_cluster_bootstrap_is_deterministic_and_respects_constant_effect(self) -> None:
        values = np.full(12, 0.125)
        clusters = np.repeat(["2025-01-01", "2025-01-02", "2025-01-03"], 4)
        first = cluster_bootstrap_median_interval(values, clusters, "constant", reps=99)
        second = cluster_bootstrap_median_interval(values, clusters, "constant", reps=99)
        self.assertEqual(first, second)
        self.assertEqual(first, (0.125, 0.125))


if __name__ == "__main__":
    unittest.main()
