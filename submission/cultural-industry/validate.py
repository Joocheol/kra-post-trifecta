#!/usr/bin/env python3
import json
import subprocess
import sys
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
    "sample": "19,284",
    "clean": "3,321",
    "superiority": "97.6%",
    "capped_only": "[0.0572, 0.0727]",
    "external_exacta": "0.0275[0.0062, 0.0486]",
    "external_quinella": "0.0235[0.0026, 0.0436]",
    "external_trio": "0.0423[0.0196, 0.0665]",
    "turnover_observed": "turnover_won",
    "law": "「한국마사회법 시행령」 제11조",
    "outer_bound": "보수적 상한(outer bound)",
}
for label, needle in checks.items():
    assert needle in text, f"missing {label}: {needle}"

for forbidden in ["정보효율성", "엄격한 강한", "가격 보정기울기", "공개 GitHub 저장소"]:
    assert forbidden not in text, f"forbidden legacy wording: {forbidden}"

blind = subprocess.check_output(["pdftotext", str(required[3]), "-"]).decode("utf-8", errors="ignore")
for term in ["김주철", "Joocheol", "연세대학교", "yonsei"]:
    assert term not in blind, f"blind PDF leaks identity: {term}"

info = subprocess.check_output(["pdfinfo", str(required[2])]).decode("utf-8", errors="ignore")
assert "Page size:       612 x 792 pts (letter)" in info, "review PDF is not letter size"
assert "Pages:           10" in info, "review PDF is not 10 pages"

report = json.loads((HERE / "hwp-build-report.json").read_text(encoding="utf-8"))
for output in report["outputs"]:
    assert output["loss"]["count"] == 0, output
    assert output["verify"]["recovered"] is True, output
    assert output["verify"]["pageCountBefore"] == output["verify"]["pageCountAfter"], output

print("submission validation passed")
