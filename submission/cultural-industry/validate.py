#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "manuscript.json").read_text(encoding="utf-8"))

required = [
    HERE / "kra-cultural-industry-submission-author.hwp",
    HERE / "kra-cultural-industry-submission-blind.hwp",
    HERE / "kra-cultural-industry-submission-author-review.pdf",
    HERE / "kra-cultural-industry-submission-blind-review.pdf",
]
for f in required:
    assert f.exists() and f.stat().st_size > 10_000, f"missing or too small: {f}"

text = json.dumps(data, ensure_ascii=False)
checks = {
    "sample": "19,301",
    "clean": "3,338",
    "superiority": "97.6%",
    "capped_only": "[0.0572, 0.0727]",
    "external_exacta": "0.0276 [0.0068, 0.0480]",
    "external_quinella": "0.0233 [0.0026, 0.0435]",
    "external_trio": "0.0421 [0.0192, 0.0651]",
    "turnover_observed": "turnover_won",
    "turnover_table": "<표 4> 실제 매출액 기반 유한풀 잡음",
    "logloss_table": "<표 5> 실제 착순 로그손실의 Harville 대비 개선",
    "outer_bound": "보수적 상한(outer bound)",
    "reader_friendly_tv": "이름은 다소 수리적으로 보이지만 해석은 간단하다",
    "more_input_not_automatic": "삼쌍승은 단승보다 더 많은 가격을 사용하므로 잘 맞는 것이 당연하다는 반론",
    "finite_liquidity": "손익분기 유효단위의 중앙값은 쌍승 56,368원",
}
for label, needle in checks.items():
    assert needle in text, f"missing {label}: {needle}"

for forbidden in [
    "정보효율성",
    "엄격한 강한",
    "가격 보정기울기",
    "공개 GitHub 저장소",
    "q[r,c]",
    "A[r]^m",
    "Σc|",
    "19,284",
    "3,321",
    "<표 4> 단계조정 Harville",
]:
    assert forbidden not in text, f"forbidden legacy or overly technical wording: {forbidden}"

blind = subprocess.check_output(["pdftotext", str(required[3]), "-"]).decode(
    "utf-8", errors="ignore"
)
for term in ["김주철", "Joocheol", "연세대학교", "yonsei"]:
    assert term not in blind, f"blind PDF leaks identity: {term}"

info = subprocess.check_output(["pdfinfo", str(required[2])]).decode(
    "utf-8", errors="ignore"
)
assert "Page size:       612 x 792 pts (letter)" in info, "review PDF is not letter size"
pages = next(
    int(line.split(":", 1)[1].strip())
    for line in info.splitlines()
    if line.startswith("Pages:")
)
assert 8 <= pages <= 10, f"review PDF must be 8–10 pages, got {pages}"

report = json.loads((HERE / "hwp-build-report.json").read_text(encoding="utf-8"))
for output in report["outputs"]:
    assert output["loss"]["count"] == 0, output
    assert output["verify"]["recovered"] is True, output
    assert output["verify"]["pageCountBefore"] == output["verify"]["pageCountAfter"], output

print("submission validation passed")
