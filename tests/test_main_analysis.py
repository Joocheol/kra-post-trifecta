from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analysis.data_audit import TARGET_MARKETS
from analysis.display_precision_audit import off_tenth_grid_mask
from analysis.main_analysis import (
    PriceSet,
    aggregate_price_set,
    harville_trifecta,
    normalize_inverse_odds,
    odds_to_price_set,
    point_metrics,
    point_price_set,
    price_set_component_bounds,
    source_group_index,
    tv_lower_exact,
    tv_upper_outer,
)
from analysis.main_analysis_core import aggregate_point, choose_other_race
from analysis.main_analysis_guards import assert_win_uncapped
from analysis.main_analysis_p3 import order_information_bounds
from analysis.main_analysis_p3_joint import joint_p3_extrema
from analysis.main_analysis_runner import common_race_ids
from analysis.main_analysis_report import external_log_score_summary


class PriceSetTest(unittest.TestCase):
    def test_point_sets_have_zero_tv(self) -> None:
        p = np.array([0.2, 0.3, 0.5])
        ps = point_price_set(p)
        self.assertAlmostEqual(tv_lower_exact(ps, ps), 0.0, places=10)
        self.assertAlmostEqual(tv_upper_outer(ps, ps), 0.0, places=10)

    def test_disjoint_singletons_match_direct_tv(self) -> None:
        p = np.array([0.8, 0.2])
        q = np.array([0.3, 0.7])
        direct = 0.5 * np.abs(p - q).sum()
        self.assertAlmostEqual(
            tv_lower_exact(point_price_set(p), point_price_set(q)), direct, places=10
        )
        self.assertAlmostEqual(
            tv_upper_outer(point_price_set(p), point_price_set(q)), direct, places=10
        )

    def test_interval_contains_display_point(self) -> None:
        odds = np.array([2.0, 4.0, 9999.9])
        capped = np.array([False, False, True])
        ps = odds_to_price_set(odds, capped)
        p = normalize_inverse_odds(odds)
        pmin, pmax = price_set_component_bounds(ps)
        self.assertTrue(np.all(p >= pmin - 1e-12))
        self.assertTrue(np.all(p <= pmax + 1e-12))
        self.assertEqual(ps.lower[-1], 0.0)

    def test_aggregation_preserves_partition_mass(self) -> None:
        ps = PriceSet(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        )
        agg = aggregate_price_set(ps, np.array([0, 0, 1, 1]), 2)
        p = agg.lower / agg.lower.sum()
        np.testing.assert_allclose(p, [0.3, 0.7])

    def test_outer_upper_bound_contains_sampled_feasible_tv(self) -> None:
        left = PriceSet(np.array([0.2, 0.1, 0.05]), np.array([0.3, 0.2, 0.15]))
        right = PriceSet(np.array([0.1, 0.15, 0.05]), np.array([0.25, 0.3, 0.2]))
        lower = tv_lower_exact(left, right)
        upper = tv_upper_outer(left, right)
        self.assertLessEqual(lower, upper + 1e-10)
        rng = np.random.default_rng(7)
        for _ in range(250):
            x = rng.uniform(left.lower, left.upper)
            y = rng.uniform(right.lower, right.upper)
            p = x / x.sum()
            q = y / y.sum()
            tv = 0.5 * np.abs(p - q).sum()
            self.assertGreaterEqual(tv + 1e-9, lower)
            self.assertLessEqual(tv, upper + 1e-9)


class DisplayPrecisionTest(unittest.TestCase):
    def test_uncapped_tenth_grid_detection(self) -> None:
        odds = np.array([1.0, 2.3, 99.9, 1000.0, 9999.9, 2.35])
        capped = np.array([False, False, False, False, True, False])
        mask = off_tenth_grid_mask(odds, capped)
        np.testing.assert_array_equal(mask, [False, False, False, False, False, True])


class GuardTest(unittest.TestCase):
    def test_win_cap_guard_rejects_capped_observation(self) -> None:
        frame = pd.DataFrame({"is_capped_odds": [False, True]})
        with self.assertRaisesRegex(ValueError, "requires uncapped win odds"):
            assert_win_uncapped(frame)

    def test_win_cap_guard_accepts_uncapped_observations(self) -> None:
        assert_win_uncapped(pd.DataFrame({"is_capped_odds": [False, False]}))


class MappingTest(unittest.TestCase):
    @staticmethod
    def _trifecta_frame(horses: list[int]) -> pd.DataFrame:
        rows = []
        for i in horses:
            for j in horses:
                for k in horses:
                    if len({i, j, k}) == 3:
                        rows.append((i, j, k))
        return pd.DataFrame(
            sorted(rows), columns=["first_no", "second_no", "third_no"]
        )

    def test_trifecta_to_targets(self) -> None:
        source = pd.DataFrame(
            {
                "first_no": [1, 1, 2, 2, 3, 3],
                "second_no": [2, 3, 1, 3, 1, 2],
                "third_no": [3, 2, 3, 1, 2, 1],
            }
        )
        exacta = pd.DataFrame(
            {
                "first_no": [1, 1, 2, 2, 3, 3],
                "second_no": [2, 3, 1, 3, 1, 2],
            }
        )
        groups = source_group_index(source, exacta, "exacta")
        np.testing.assert_array_equal(groups, np.arange(6))
        quinella = pd.DataFrame(
            {"horse_a": [1, 1, 2], "horse_b": [2, 3, 3]}
        )
        groups = source_group_index(source, quinella, "quinella")
        self.assertEqual(set(groups.tolist()), {0, 1, 2})

    def test_harville_sums_to_one(self) -> None:
        horses = [1, 2, 3, 4]
        source = self._trifecta_frame(horses)
        win = pd.DataFrame(
            {
                "horse_no": horses,
                "odds": [2.0, 3.0, 4.0, 5.0],
                "is_capped_odds": [False, False, False, False],
            }
        )
        q = harville_trifecta(source, win)
        self.assertAlmostEqual(float(q.sum()), 1.0, places=12)
        self.assertTrue(np.all(q > 0))

    def test_other_race_transfer_is_rank_aligned_with_gapped_horse_numbers(self) -> None:
        anchor_horses = [1, 3, 7, 10]
        donor_horses = [2, 5, 9, 14]
        anchor_source = self._trifecta_frame(anchor_horses)
        donor_source = self._trifecta_frame(donor_horses)
        anchor_exacta = pd.DataFrame(
            sorted((i, j) for i in anchor_horses for j in anchor_horses if i != j),
            columns=["first_no", "second_no"],
        )
        donor_exacta = pd.DataFrame(
            sorted((i, j) for i in donor_horses for j in donor_horses if i != j),
            columns=["first_no", "second_no"],
        )
        anchor_groups = source_group_index(anchor_source, anchor_exacta, "exacta")
        donor_groups = source_group_index(donor_source, donor_exacta, "exacta")
        np.testing.assert_array_equal(anchor_groups, donor_groups)

        chosen = choose_other_race("anchor", ["anchor", "donor"], "B")
        self.assertEqual(chosen, "donor")
        donor_weights = np.zeros(len(donor_source))
        donor_weights[0] = 1.0
        transferred = aggregate_point(
            donor_weights, anchor_groups, len(anchor_exacta)
        )
        donor_native = aggregate_point(
            donor_weights, donor_groups, len(donor_exacta)
        )
        np.testing.assert_allclose(transferred, donor_native)
        self.assertEqual(int(np.argmax(transferred)), 0)


class MetricTest(unittest.TestCase):
    def test_identity_metrics(self) -> None:
        p = np.array([0.1, 0.2, 0.7])
        metrics = point_metrics(p, p)
        self.assertAlmostEqual(metrics["tv"], 0.0)
        self.assertAlmostEqual(metrics["js"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)


class FrozenSampleTest(unittest.TestCase):
    @staticmethod
    def _sample(ids_by_target: dict[str, list[str]]) -> pd.DataFrame:
        rows = []
        for target in TARGET_MARKETS:
            for race_id in ids_by_target[target]:
                rows.append(
                    {
                        "race_id": race_id,
                        "target_market": target,
                        "eligible_clean_point_sample": True,
                    }
                )
        return pd.DataFrame(rows)

    def test_common_race_ids_accepts_identical_target_sets(self) -> None:
        ids = {target: ["r1", "r2"] for target in TARGET_MARKETS}
        sample = self._sample(ids)
        self.assertEqual(
            common_race_ids(sample, "eligible_clean_point_sample"), ["r1", "r2"]
        )

    def test_common_race_ids_rejects_target_mismatch(self) -> None:
        ids = {target: ["r1", "r2"] for target in TARGET_MARKETS}
        ids["trio"] = ["r1"]
        sample = self._sample(ids)
        with self.assertRaisesRegex(ValueError, "target samples differ"):
            common_race_ids(sample, "eligible_clean_point_sample")


class OrderInformationBoundsTest(unittest.TestCase):
    def test_conservative_difference_uses_opposite_endpoints(self) -> None:
        rows = []
        values = {
            ("exacta", "harville"): (0.20, 0.22),
            ("exacta", "main"): (0.10, 0.12),
            ("quinella", "harville"): (0.16, 0.18),
            ("quinella", "main"): (0.11, 0.13),
        }
        for race_id in ("r1", "r2"):
            for (target, model), (lower, upper) in values.items():
                rows.append(
                    {
                        "race_id": race_id,
                        "target_market": target,
                        "model": model,
                        "tv_lower": lower,
                        "tv_upper_outer": upper,
                    }
                )
        result = order_information_bounds(pd.DataFrame(rows)).iloc[0]
        expected_lower = 0.20 - 0.12 - 0.18 + 0.11
        expected_upper = 0.22 - 0.10 - 0.16 + 0.13
        self.assertAlmostEqual(result["median_difference_lower"], expected_lower)
        self.assertAlmostEqual(result["median_difference_upper"], expected_upper)
        self.assertTrue(result["robust_positive_difference"])

    def test_joint_milp_collapses_to_direct_difference_for_point_sets(self) -> None:
        source = MappingTest._trifecta_frame([1, 2, 3])
        exacta = pd.DataFrame(
            sorted((i, j) for i in [1, 2, 3] for j in [1, 2, 3] if i != j),
            columns=["first_no", "second_no"],
        )
        quinella = pd.DataFrame(
            [(1, 2), (1, 3), (2, 3)], columns=["horse_a", "horse_b"]
        )
        e_groups = source_group_index(source, exacta, "exacta")
        q_groups = source_group_index(source, quinella, "quinella")
        source_p = np.array([0.20, 0.15, 0.10, 0.20, 0.15, 0.20])
        actual_e = np.array([0.18, 0.17, 0.12, 0.18, 0.16, 0.19])
        actual_q = np.array([0.31, 0.29, 0.40])
        h_source = np.array([0.17, 0.13, 0.11, 0.21, 0.16, 0.22])
        main_e = aggregate_point(source_p, e_groups, len(exacta))
        main_q = aggregate_point(source_p, q_groups, len(quinella))
        h_e = aggregate_point(h_source, e_groups, len(exacta))
        h_q = aggregate_point(h_source, q_groups, len(quinella))
        tv = lambda left, right: 0.5 * float(np.abs(left - right).sum())
        expected = (
            tv(actual_e, h_e)
            - tv(actual_e, main_e)
            - tv(actual_q, h_q)
            + tv(actual_q, main_q)
        )
        lower, upper = joint_p3_extrema(
            point_price_set(source_p),
            e_groups,
            point_price_set(actual_e),
            h_e,
            q_groups,
            point_price_set(actual_q),
            h_q,
            time_limit=10.0,
        )
        self.assertAlmostEqual(lower, expected, places=8)
        self.assertAlmostEqual(upper, expected, places=8)


class ExternalLogScoreTest(unittest.TestCase):
    def test_summary_uses_paired_races_and_positive_means_main_is_better(self) -> None:
        rows = []
        for race_id, race_date, main, harville in (
            ("r1", "2025-01-01", 1.0, 1.2),
            ("r2", "2025-01-01", 1.4, 1.5),
            ("r3", "2025-01-02", 0.9, 1.1),
        ):
            rows.extend(
                [
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "target_market": "exacta",
                        "model": "main",
                        "realized_log_score": main,
                        "realized_epsilon_bound": False,
                    },
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "target_market": "exacta",
                        "model": "harville",
                        "realized_log_score": harville,
                        "realized_epsilon_bound": False,
                    },
                ]
            )
        state_records = []
        for row in rows:
            realized_probability = float(np.exp(-float(row["realized_log_score"])))
            state_records.append(
                {
                    "race_id": row["race_id"],
                    "race_date": row["race_date"],
                    "year": 2024 if row["race_id"] == "r1" else 2025,
                    "target_market": row["target_market"],
                    "model": row["model"],
                    "predicted": np.array(
                        [realized_probability, 1.0 - realized_probability]
                    ),
                    "realized_index": 0,
                }
            )
        result = external_log_score_summary(pd.DataFrame(rows), state_records).iloc[0]
        self.assertEqual(int(result["n_races"]), 3)
        self.assertAlmostEqual(float(result["raw_mean_improvement"]), 1 / 6)
        self.assertGreater(float(result["raw_date_cluster_ci_low"]), 0.0)
        self.assertEqual(int(result["raw_main_epsilon_bound_count"]), 0)
        self.assertEqual(int(result["raw_harville_epsilon_bound_count"]), 0)


if __name__ == "__main__":
    unittest.main()
