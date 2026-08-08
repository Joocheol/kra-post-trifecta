from __future__ import annotations

import unittest

from analysis.data_audit import expected_row_count, parse_horse_list


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


if __name__ == "__main__":
    unittest.main()
