import unittest

import numpy as np
import pandas as pd

from analysis.cultural_industry_reverification import (
    canonical_cap,
    harville_on_trifecta_support,
    marginalize_source,
    tv_distance,
)


class CulturalIndustryReverificationTest(unittest.TestCase):
    def test_only_exact_9999_9_is_censored(self):
        odds = pd.Series([9999.8, 9999.9, 10000.0, 11732.4])
        self.assertEqual(canonical_cap(odds).tolist(), [False, True, False, False])

    def test_tv_zero_for_identical_vectors(self):
        idx = pd.Index([1, 2, 3], name="horse_no")
        p = pd.Series([0.5, 0.3, 0.2], index=idx)
        self.assertAlmostEqual(tv_distance(p, p.copy()), 0.0, places=15)

    def test_harville_trifecta_mass_and_marginals(self):
        horses = [1, 2, 3]
        rows = []
        for i in horses:
            for j in horses:
                for k in horses:
                    if len({i, j, k}) == 3:
                        rows.append((i, j, k))
        source = pd.DataFrame(rows, columns=["first_no", "second_no", "third_no"])
        win = pd.DataFrame(
            {
                "horse_no": horses,
                # inverse odds normalize to 0.5, 0.3, 0.2
                "odds": [2.0, 10.0 / 3.0, 5.0],
            }
        )
        h = harville_on_trifecta_support(source, win)
        self.assertAlmostEqual(float(h.sum()), 1.0, places=14)
        win_marginal = marginalize_source(source, h, "win")
        expected = pd.Series([0.5, 0.3, 0.2], index=pd.Index(horses, name="horse_no"))
        self.assertTrue(np.allclose(win_marginal.to_numpy(), expected.to_numpy(), atol=1e-14))
        exacta = marginalize_source(source, h, "exacta")
        self.assertAlmostEqual(float(exacta.sum()), 1.0, places=14)
        quinella = marginalize_source(source, h, "quinella")
        self.assertAlmostEqual(float(quinella.sum()), 1.0, places=14)
        trio = marginalize_source(source, h, "trio")
        self.assertAlmostEqual(float(trio.sum()), 1.0, places=14)
        self.assertEqual(len(trio), 1)


if __name__ == "__main__":
    unittest.main()
