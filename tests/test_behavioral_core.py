from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analysis.behavioral_core import (
    MonotoneCalibrator,
    PowerWeight,
    PrelecWeight,
    conditional_stage_probability,
    distribution_metrics,
    expected_calibration_error,
    fit_stage_temperature,
    normalized_inverse_odds,
    summarize_race_metrics,
)
from analysis.behavioral_analysis import score_price_model


class BehavioralCoreTests(unittest.TestCase):
    def test_normalized_inverse_odds_is_a_distribution(self) -> None:
        result = normalized_inverse_odds(np.array([2.0, 4.0, 8.0]))
        np.testing.assert_allclose(result, np.array([4.0, 2.0, 1.0]) / 7.0)

    def test_monotone_calibrator_pools_violations_and_tracks_support(self) -> None:
        fit = MonotoneCalibrator.fit(
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.4, 0.2, 0.8],
        )
        predicted = fit.predict([0.1, 0.2, 0.3, 0.4])
        self.assertTrue(np.all(np.diff(predicted) >= 0))
        self.assertEqual(fit.support, (0.1, 0.4))
        with self.assertRaisesRegex(ValueError, "outside"):
            fit.predict([0.05], clip=False)

    def test_stage_temperature_recovers_sharper_second_choices(self) -> None:
        rng = np.random.default_rng(20260810)
        choice_sets: list[tuple[np.ndarray, int]] = []
        true_alpha = 1.7
        for _ in range(2500):
            strength = rng.dirichlet(np.ones(8))
            probability = strength**true_alpha
            probability /= probability.sum()
            chosen = int(rng.choice(len(strength), p=probability))
            choice_sets.append((strength, chosen))
        fitted = fit_stage_temperature(choice_sets)
        self.assertAlmostEqual(float(fitted.alpha(8)), true_alpha, delta=0.15)

    def test_conditional_probability_removes_prior_choices(self) -> None:
        chosen = np.array([0.3, 0.2])
        total = np.array([1.0, 1.0])
        excluded = [np.array([0.4, 0.5])]
        result = conditional_stage_probability(chosen, total, excluded)
        np.testing.assert_allclose(result, np.array([0.5, 0.4]))

    def test_prelec_and_power_weights_are_positive_and_increasing(self) -> None:
        p = np.geomspace(0.002, 0.5, 200)
        q = np.exp(-0.8 * (-np.log(p)) ** 0.7)
        prelec = PrelecWeight.fit(p, q)
        self.assertAlmostEqual(prelec.alpha, 0.7, delta=0.03)
        self.assertAlmostEqual(prelec.beta, 0.8, delta=0.03)
        power = PowerWeight.fit(p, p**0.75)
        self.assertAlmostEqual(power.exponent, 0.75, places=10)
        self.assertTrue(np.all(np.diff(prelec.predict(p)) > 0))
        self.assertTrue(np.all(np.diff(power.predict(p)) > 0))

    def test_power_weight_is_reduction_invariant_in_deep_tail(self) -> None:
        frame = pd.DataFrame(
            {
                "p1": [0.3, 0.002],
                "p2_cond": [0.2, 0.0002],
                "p3_cond": [0.1, 0.00002],
            }
        )
        frame["p_joint"] = frame["p1"] * frame["p2_cond"] * frame["p3_cond"]
        predictor = PowerWeight(0.6).predict
        reduced, _ = score_price_model(frame, "trifecta", "M-R", predictor)
        sequential2, _ = score_price_model(frame, "trifecta", "M-S2", predictor)
        sequential3, _ = score_price_model(frame, "trifecta", "M-S3", predictor)
        np.testing.assert_allclose(reduced, sequential2, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(reduced, sequential3, rtol=1e-13, atol=0.0)

    def test_distribution_metrics_vanish_for_identical_vectors(self) -> None:
        probability = np.array([0.2, 0.3, 0.5])
        result = distribution_metrics(probability, probability)
        for value in result.values():
            self.assertAlmostEqual(value, 0.0)

    def test_expected_calibration_error_is_zero_for_exact_bins(self) -> None:
        probability = np.array([0.0, 0.0, 1.0, 1.0])
        outcome = probability.copy()
        self.assertAlmostEqual(
            expected_calibration_error(probability, outcome, bins=2), 0.0
        )

    def test_metric_summary_uses_race_equal_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "validation_year": [2024, 2025],
                "probability_model": ["harville", "harville"],
                "tail_model": ["prelec", "prelec"],
                "price_model": ["M-R", "M-R"],
                "target_market": ["exacta", "exacta"],
                "race_id": ["a", "b"],
                "tv": [0.1, 0.3],
                "mae": [0.01, 0.03],
                "log_rmse": [0.2, 0.4],
                "js": [0.02, 0.06],
                "support_share": [1.0, 0.5],
            }
        )
        result = summarize_race_metrics(frame).iloc[0]
        self.assertEqual(int(result["n_races"]), 2)
        self.assertAlmostEqual(float(result["median_tv"]), 0.2)
        self.assertAlmostEqual(float(result["mean_support_share"]), 0.75)


if __name__ == "__main__":
    unittest.main()
