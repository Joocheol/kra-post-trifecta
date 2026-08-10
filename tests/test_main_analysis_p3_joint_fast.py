from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analysis.main_analysis_core import (
    aggregate_point,
    point_price_set,
    source_group_index,
)
from analysis.main_analysis_p3_joint_fast import joint_p3_certified


class TightJointP3Test(unittest.TestCase):
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

    def test_point_sets_return_sharp_direct_difference(self) -> None:
        horses = [1, 2, 3]
        source = self._trifecta_frame(horses)
        exacta = pd.DataFrame(
            sorted((i, j) for i in horses for j in horses if i != j),
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
        lower, upper, minimum, maximum = joint_p3_certified(
            point_price_set(source_p),
            e_groups,
            point_price_set(actual_e),
            h_e,
            q_groups,
            point_price_set(actual_q),
            h_q,
            time_limit=10.0,
        )
        self.assertTrue(minimum.optimal)
        self.assertTrue(maximum.optimal)
        self.assertAlmostEqual(lower, expected, places=8)
        self.assertAlmostEqual(upper, expected, places=8)


if __name__ == "__main__":
    unittest.main()
