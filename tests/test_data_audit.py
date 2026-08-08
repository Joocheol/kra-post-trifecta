from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.data_audit import (
    TARGET_MARKETS,
    allowed_horse_matrix,
    audit_market,
    build_analysis_sample,
    clean_sample_set_differences,
    expected_row_count,
    invalid_key_mask,
    parse_horse_list,
    structural_masks,
)


class ExpectedRowCountTest(unittest.TestCase):
    def test_ten_horse_support_sizes(self) -> None:
        self.assertEqual(expected_row_count("win", 10), 10)
        self.assertEqual(expected_row_count("exacta", 10), 90)
        self.assertEqual(expected_row_count("quinella", 10), 45)
        self.assertEqual(expected_row_count("trio", 10), 120)
        self.assertEqual(expected_row_count("trifecta", 10), 720)

    def test_scratched_horse_field_uses_count_not_max_number(self) -> None:
        horses = parse_horse_list("1,2,3,4,6,7,8,9,10")
        self.assertEqual(len(horses), 9)
        self.assertEqual(expected_row_count("trifecta", len(horses)), 504)


class ParseHorseListTest(unittest.TestCase):
    def test_empty_and_malformed_values_are_empty(self) -> None:
        self.assertEqual(parse_horse_list(""), ())
        self.assertEqual(parse_horse_list(None), ())
        self.assertEqual(parse_horse_list("1,x,3"), ())

    def test_valid_list_preserves_numbers(self) -> None:
        self.assertEqual(parse_horse_list("2,3,5"), (2, 3, 5))


class KeyValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.races = pd.DataFrame(
            {"race_id": ["r1"], "valid_horse_tuple": [(1, 2, 3)]}
        )
        self.race_codes, self.allowed = allowed_horse_matrix(self.races)

    def test_ordered_keys_reject_duplicates_out_of_support_and_orphans(self) -> None:
        frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r1", "orphan"],
                "first_no": [1, 1, 1, 1],
                "second_no": [2, 1, 4, 2],
            }
        )
        result = invalid_key_mask(
            frame, "exacta", self.race_codes, self.allowed
        ).tolist()
        self.assertEqual(result, [False, True, True, True])

    def test_unordered_keys_require_canonical_order(self) -> None:
        frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1"],
                "horse_a": [1, 2],
                "horse_b": [2, 1],
            }
        )
        result = invalid_key_mask(
            frame, "quinella", self.race_codes, self.allowed
        ).tolist()
        self.assertEqual(result, [False, True])


class MarketAuditTest(unittest.TestCase):
    def test_status_caps_and_orphan_ids_are_checked(self) -> None:
        races = pd.DataFrame(
            {
                "race_id": ["r1", "r2", "r3"],
                "n_valid_horses": [3, 3, 3],
                "valid_horse_tuple": [(1, 2, 3)] * 3,
                "race_date": ["2025-01-01"] * 3,
                "meet": ["1"] * 3,
            }
        )
        status = pd.DataFrame(
            {
                "race_id": ["r1", "r2", "r3"],
                "market": ["win"] * 3,
                "n_rows": [3, 3, 3],
                "status": ["ok"] * 3,
                "is_cancelled": [False] * 3,
                "status_reason": ["parsed_rows_present"] * 3,
            }
        ).set_index(["race_id", "market"])
        market = pd.DataFrame(
            {
                "race_id": [
                    "r1",
                    "r1",
                    "r1",
                    "r2",
                    "r2",
                    "r2",
                    "r3",
                    "r3",
                    "r3",
                    "orphan",
                ],
                "horse_no": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
                "odds": [
                    2.0,
                    3.0,
                    4.0,
                    9999.9,
                    3.0,
                    4.0,
                    9999.9,
                    3.0,
                    4.0,
                    2.0,
                ],
                "is_capped_odds": [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                "race_date": ["2025-01-01"] * 10,
                "meet": ["1"] * 10,
            }
        )
        race_codes, allowed = allowed_horse_matrix(races)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = (
                root
                / "kra_test"
                / "market=win"
                / "year=2025"
                / "month=2025-01"
                / "part-0.parquet"
            )
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pandas(market, preserve_index=False), path)
            result = audit_market(root, "win", races, status, race_codes, allowed)

        self.assertTrue(bool(result.loc["r1", "win_complete_support"]))
        self.assertTrue(bool(result.loc["r1", "win_positive_finite_odds"]))
        self.assertFalse(bool(result.loc["r2", "win_uncapped"]))
        self.assertTrue(bool(result.loc["r2", "win_positive_finite_odds"]))
        self.assertFalse(bool(result.loc["r3", "win_positive_finite_odds"]))
        self.assertEqual(int(result.loc["r1", "win_orphan_race_ids_detected"]), 1)
        self.assertEqual(int(result.loc["r1", "win_orphan_rows_detected"]), 1)


class SampleConstructionTest(unittest.TestCase):
    def test_clean_interval_and_structural_exclusion_are_disjoint(self) -> None:
        quality = pd.DataFrame(
            {
                "race_id": ["clean", "capped", "incomplete"],
                "race_date": ["2025-01-01"] * 3,
                "meet": ["1"] * 3,
                "race_no": [1, 2, 3],
                "n_valid_horses": [3, 3, 3],
                "in_date_scope": [True, True, True],
                "scope_exclusion_reason": ["", "", ""],
                "valid_horses_ok": [True, True, True],
                "trifecta_complete_support": [True, True, True],
                "trifecta_positive_finite_odds": [True, True, True],
                "trifecta_uncapped": [True, False, True],
                "trifecta_capped_odds_rows": [0, 1, 0],
            }
        )
        for target in TARGET_MARKETS:
            quality[f"{target}_complete_support"] = [True, True, False]
            quality[f"{target}_positive_finite_odds"] = [True, True, True]
            quality[f"{target}_uncapped"] = [True, True, True]
            quality[f"{target}_capped_odds_rows"] = [0, 0, 0]

        masks = structural_masks(quality, "exacta")
        self.assertEqual(
            masks["exacta_positive_finite_odds"].tolist(),
            [True, True, False],
        )

        sample = build_analysis_sample(quality)
        exacta = sample[sample["target_market"].eq("exacta")].set_index("race_id")
        self.assertTrue(bool(exacta.loc["clean", "eligible_clean_point_sample"]))
        self.assertTrue(bool(exacta.loc["capped", "eligible_capped_interval_sample"]))
        self.assertFalse(bool(exacta.loc["incomplete", "eligible_complete_sample"]))
        self.assertEqual(
            exacta.loc["incomplete", "structural_exclusion_reason"],
            "incomplete_exacta_support",
        )
        self.assertFalse(any(clean_sample_set_differences(sample).values()))

    def test_clean_race_id_mismatch_is_detected_even_when_counts_match(self) -> None:
        sample = pd.DataFrame(
            {
                "target_market": [
                    "win",
                    "win",
                    "exacta",
                    "exacta",
                    "quinella",
                    "quinella",
                    "trio",
                    "trio",
                ],
                "race_id": ["r1", "r2", "r1", "r3", "r1", "r2", "r1", "r2"],
                "eligible_clean_point_sample": [True] * 8,
            }
        )
        differences = clean_sample_set_differences(sample)
        self.assertEqual(differences["win"], 0)
        self.assertEqual(differences["exacta"], 2)
        self.assertEqual(differences["quinella"], 0)
        self.assertEqual(differences["trio"], 0)


if __name__ == "__main__":
    unittest.main()
