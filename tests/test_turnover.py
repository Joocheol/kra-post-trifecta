from __future__ import annotations

import unittest

from analysis.turnover import extract_turnovers


class TurnoverParserTests(unittest.TestCase):
    def test_extracts_every_market_and_keeps_scm_totals_separate(self) -> None:
        pages = {
            "Scm": """
                <tfoot><tr>
                <th>단승식 : 28,752,200원</th>
                <th>연승식 : 34,560,800원</th>
                <th>복승식 : 722,238,100원</th>
                <th>총매출액 : 2,479,449,800원</th>
                </tr></tfoot>
            """,
            "Both": "<th>쌍승식 매출총액</th><th>250,537,800원</th>",
            "Bc": "<th>복연승식 매출총액</th><th>111,895,300원</th>",
            "3Bc": {
                "_probe": "<th>삼복승식 매출총액</th><th>831,957,500원</th>"
            },
            "3Both": {
                "_probe": "<th>삼쌍승식 매출총액</th><th>499,508,100원</th>"
            },
        }

        self.assertEqual(
            extract_turnovers(pages),
            {
                "win": 28_752_200,
                "place": 34_560_800,
                "quinella": 722_238_100,
                "exacta": 250_537_800,
                "quinella_place": 111_895_300,
                "trio": 831_957_500,
                "trifecta": 499_508_100,
            },
        )

    def test_missing_page_is_explicitly_missing(self) -> None:
        result = extract_turnovers({"Both": "<p>취소된 경주</p>"})
        self.assertTrue(all(value is None for value in result.values()))


if __name__ == "__main__":
    unittest.main()
