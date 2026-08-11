from __future__ import annotations

import unittest

import pandas as pd

from analysis.verify_behavioral_outputs import compare_frames


class VerifyBehavioralOutputsTest(unittest.TestCase):
    def test_accepts_small_float_drift_but_not_integer_or_label_changes(self) -> None:
        expected = pd.DataFrame(
            {"label": ["a", "b"], "n_races": [10, 20], "metric": [0.1, 0.2]}
        )
        close = expected.copy()
        close["metric"] += 1e-6
        compare_frames(expected, close, "example.csv")

        bad_integer = expected.copy()
        bad_integer.loc[0, "n_races"] = 11
        with self.assertRaisesRegex(AssertionError, "integer column changed"):
            compare_frames(expected, bad_integer, "example.csv")

        bad_label = expected.copy()
        bad_label.loc[0, "label"] = "z"
        with self.assertRaisesRegex(AssertionError, "categorical column changed"):
            compare_frames(expected, bad_label, "example.csv")

    def test_rejects_material_float_change(self) -> None:
        expected = pd.DataFrame({"metric": [0.1]})
        actual = pd.DataFrame({"metric": [0.101]})
        with self.assertRaises(AssertionError):
            compare_frames(expected, actual, "example.csv")


if __name__ == "__main__":
    unittest.main()
