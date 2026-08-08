from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

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


class MappingTest(unittest.TestCase):
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
        rows = []
        for i in horses:
            for j in horses:
                for k in horses:
                    if len({i, j, k}) == 3:
                        rows.append((i, j, k))
        source = pd.DataFrame(
            rows, columns=["first_no", "second_no", "third_no"]
        )
        win = pd.DataFrame(
            {"horse_no": horses, "odds": [2.0, 3.0, 4.0, 5.0]}
        )
        q = harville_trifecta(source, win)
        self.assertAlmostEqual(float(q.sum()), 1.0, places=12)
        self.assertTrue(np.all(q > 0))


class MetricTest(unittest.TestCase):
    def test_identity_metrics(self) -> None:
        p = np.array([0.1, 0.2, 0.7])
        metrics = point_metrics(p, p)
        self.assertAlmostEqual(metrics["tv"], 0.0)
        self.assertAlmostEqual(metrics["js"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()
