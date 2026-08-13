from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CovidTurnoverLiveCollectionTest(unittest.TestCase):
    """One-shot live collection used for the 2020-2021 referee diagnostic.

    This test is intentionally temporary. It exercises the public KRA archive from
    GitHub Actions, where outbound network access is available, and prints the compact
    summary into the CI log so the diagnostic can be inspected before freezing it.
    Non-race calendar dates are expected to return quickly; short timeouts keep this
    one-shot scan bounded without relying on assumptions about racing weekdays.
    """

    def test_collect_2020_2021_race1_turnover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = tmp_path / "covid_turnover_race1.csv"
            summary = tmp_path / "covid_turnover_race1_summary.csv"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "analysis.covid_turnover_collection",
                    "--start-date",
                    "2020-01-01",
                    "--end-date",
                    "2021-12-31",
                    "--workers",
                    "16",
                    "--timeout",
                    "5",
                    "--retries",
                    "0",
                    "--output",
                    str(rows),
                    "--summary-output",
                    str(summary),
                ],
                check=True,
                timeout=780,
            )
            with rows.open(encoding="utf-8") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
            self.assertGreater(row_count, 0)
            print("COVID_TURNOVER_SUMMARY_BEGIN")
            print(summary.read_text(encoding="utf-8"), end="")
            print("COVID_TURNOVER_SUMMARY_END")


if __name__ == "__main__":
    unittest.main()
