from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.main_analysis_p3_joint_fast import write_table


class P3FrozenTableTest(unittest.TestCase):
    def test_frozen_table_matches_current_generator(self) -> None:
        """Prevent a stale hand-edited P3 LaTeX table from passing CI."""
        frame = pd.read_csv("outputs/main_order_information_joint.csv")
        frozen = Path("tables/main_order_information_joint.tex").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            generated_path = Path(tmp) / "main_order_information_joint.tex"
            write_table(frame, generated_path)
            generated = generated_path.read_text(encoding="utf-8")
        self.assertEqual(
            frozen,
            generated,
            "frozen P3 LaTeX table differs from main_analysis_p3_joint_fast.write_table()",
        )


if __name__ == "__main__":
    unittest.main()
