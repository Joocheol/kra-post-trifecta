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


if __name__ == "__main__":
    unittest.main()
