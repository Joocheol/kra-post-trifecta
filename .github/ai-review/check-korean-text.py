#!/usr/bin/env python3
"""Compare Hangul recovered from the PDF with Hangul in manuscript sources."""

from __future__ import annotations

import sys
from pathlib import Path


def hangul_count(text: str) -> int:
    return sum(
        1
        for char in text
        if "\uac00" <= char <= "\ud7a3" or "\u1100" <= char <= "\u11ff"
    )


def source_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.tex")))
        elif path.is_file():
            files.append(path)
    return files


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: check-korean-text.py EXTRACTED_TEXT SOURCE...", file=sys.stderr)
        return 2

    extracted_path = Path(sys.argv[1])
    sources = source_files(sys.argv[2:])
    if not extracted_path.is_file() or not sources:
        print("FAIL: extracted text or manuscript sources are missing", file=sys.stderr)
        return 1

    extracted = hangul_count(extracted_path.read_text(encoding="utf-8", errors="replace"))
    source = sum(hangul_count(path.read_text(encoding="utf-8")) for path in sources)
    ratio = extracted / source if source else 0.0

    print(f"source_hangul={source}")
    print(f"extracted_hangul={extracted}")
    print(f"hangul_recovery_ratio={ratio:.4f}")
    print("hangul_recovery_required=0.7000")

    ok = source >= 200 and extracted >= 200 and ratio >= 0.70
    print(f"korean_text_check={'pass' if ok else 'fail'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
