from __future__ import annotations

import unittest

import numpy as np

from analysis.main_analysis_core import PriceSet, tv_lower_exact
from analysis.main_analysis_fast import price_sets_intersect, tv_lower_exact_fast


class FastLowerBoundTest(unittest.TestCase):
    def test_intersection_returns_exact_zero(self) -> None:
        left = PriceSet(np.array([0.20, 0.10]), np.array([0.30, 0.20]))
        right = PriceSet(np.array([0.15, 0.12]), np.array([0.28, 0.22]))
        self.assertTrue(price_sets_intersect(left, right))
        self.assertEqual(tv_lower_exact_fast(left, right), 0.0)
        self.assertAlmostEqual(tv_lower_exact(left, right), 0.0, places=10)

    def test_disjoint_sets_fall_back_to_lp(self) -> None:
        left = PriceSet(np.array([0.8, 0.2]), np.array([0.8, 0.2]))
        right = PriceSet(np.array([0.3, 0.7]), np.array([0.3, 0.7]))
        self.assertFalse(price_sets_intersect(left, right))
        self.assertAlmostEqual(
            tv_lower_exact_fast(left, right), tv_lower_exact(left, right), places=10
        )

    def test_randomized_realistic_dimensions_match_lp(self) -> None:
        """Differentially validate the intersection shortcut beyond toy dimensions."""
        rng = np.random.default_rng(20260810)
        # 560 equals the 16-horse trio outcome count in the frozen sample.
        for dimension, replicates in ((6, 4), (30, 4), (90, 4), (180, 4), (560, 2)):
            for replicate in range(replicates):
                probability = rng.dirichlet(np.ones(dimension))
                left_scale = rng.uniform(0.7, 1.3)
                right_scale = rng.uniform(0.7, 1.3)
                left_center = left_scale * probability
                if replicate % 2 == 0:
                    # Both raw boxes contain differently scaled versions of the
                    # same normalized vector, so the exact distance is zero.
                    right_center = right_scale * probability
                else:
                    right_center = right_scale * rng.dirichlet(np.ones(dimension))
                left_width = rng.uniform(0.001, 0.08, size=dimension)
                right_width = rng.uniform(0.001, 0.08, size=dimension)
                left = PriceSet(
                    left_center * (1.0 - left_width),
                    left_center * (1.0 + left_width),
                )
                right = PriceSet(
                    right_center * (1.0 - right_width),
                    right_center * (1.0 + right_width),
                )
                with self.subTest(dimension=dimension, replicate=replicate):
                    self.assertAlmostEqual(
                        tv_lower_exact_fast(left, right),
                        tv_lower_exact(left, right),
                        places=9,
                    )


if __name__ == "__main__":
    unittest.main()
