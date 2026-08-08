#!/usr/bin/env python3
"""Verify that every PDF page produced a nonblank PNG render."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

MIN_NONWHITE = 1000
NEAR_WHITE = 245


def ink(path: Path) -> int:
    with Image.open(path) as image:
        return sum(image.convert("L").histogram()[:NEAR_WHITE])


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check-render.py PAGE_INDEX_FILE", file=sys.stderr)
        return 2

    index = Path(sys.argv[1])
    pages = [Path(line.strip()) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not pages:
        print("FAIL: page index is empty", file=sys.stderr)
        return 1

    missing = [str(path) for path in pages if not path.is_file()]
    blank = [(str(path), ink(path)) for path in pages if path.is_file() and ink(path) < MIN_NONWHITE]

    print(f"indexed_pages={len(pages)}")
    print(f"missing_pages={len(missing)}")
    print(f"blank_pages={len(blank)}")
    print("visual_review_scope=all pages")
    for path in missing:
        print(f"missing_page={path}")
    for path, count in blank:
        print(f"blank_page={path} nonwhite_pixels={count}")

    ok = not missing and not blank
    print(f"render_check={'pass' if ok else 'fail'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
