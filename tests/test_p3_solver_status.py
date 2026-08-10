from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis.main_analysis_core import PriceSet
from analysis.main_analysis_p3_joint_fast import (
    _row_solution_status,
    joint_p3_certified,
)


class P3SolverStatusTest(unittest.TestCase):
    def _args(self):
        source = PriceSet(np.array([1.0]), np.array([1.0]))
        actual = PriceSet(np.array([1.0]), np.array([1.0]))
        groups = np.array([0], dtype=np.int64)
        harville = np.array([1.0])
        return source, groups, actual, harville, groups, actual, harville

    def test_non_limit_solver_failure_is_fatal(self) -> None:
        result = SimpleNamespace(
            success=False,
            status=2,
            message="infeasible",
            fun=np.nan,
            mip_dual_bound=np.nan,
            mip_gap=np.nan,
        )
        with patch("analysis.main_analysis_p3_joint_fast.milp", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "status 2"):
                joint_p3_certified(*self._args())

    def test_limit_without_finite_dual_bound_is_fatal(self) -> None:
        result = SimpleNamespace(
            success=False,
            status=1,
            message="time limit reached",
            fun=0.0,
            mip_dual_bound=np.nan,
            mip_gap=1.0,
        )
        with patch("analysis.main_analysis_p3_joint_fast.milp", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "without a finite dual bound"):
                joint_p3_certified(*self._args())

    def test_limit_with_finite_dual_bound_certifies_both_directions(self) -> None:
        minimum = SimpleNamespace(
            success=False,
            status=1,
            message="time limit reached",
            fun=-0.10,
            mip_dual_bound=-0.25,
            mip_gap=0.60,
        )
        maximum = SimpleNamespace(
            success=False,
            status=1,
            message="time limit reached",
            fun=-0.50,
            mip_dual_bound=-0.75,
            mip_gap=0.40,
        )
        with patch(
            "analysis.main_analysis_p3_joint_fast.milp",
            side_effect=[minimum, maximum],
        ):
            lower, upper, min_result, max_result = joint_p3_certified(*self._args())

        self.assertEqual(lower, -0.25)
        self.assertEqual(upper, 0.75)
        self.assertEqual(min_result.certified_value, -0.25)
        self.assertEqual(min_result.incumbent_value, -0.10)
        self.assertFalse(min_result.optimal)
        self.assertEqual(max_result.certified_value, 0.75)
        self.assertEqual(max_result.incumbent_value, 0.50)
        self.assertFalse(max_result.optimal)

    def test_solution_status_preserves_endpoint_direction(self) -> None:
        self.assertEqual(
            _row_solution_status(SimpleNamespace(joint_min_optimal=True, joint_max_optimal=True)),
            "sharp",
        )
        self.assertEqual(
            _row_solution_status(SimpleNamespace(joint_min_optimal=False, joint_max_optimal=True)),
            "lower-certified",
        )
        self.assertEqual(
            _row_solution_status(SimpleNamespace(joint_min_optimal=True, joint_max_optimal=False)),
            "upper-certified",
        )
        self.assertEqual(
            _row_solution_status(SimpleNamespace(joint_min_optimal=False, joint_max_optimal=False)),
            "both-certified",
        )


if __name__ == "__main__":
    unittest.main()
