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
