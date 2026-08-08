#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

required=(
  main.tex
  preamble.tex
  references.bib
  RESEARCH_PLAN.md
  WRITING_GUIDE.md
  CLAUDE.md
)

for path in "${required[@]}"; do
  test -s "$path" || {
    echo "FAIL: required file is missing or empty: $path" >&2
    exit 1
  }
done

git diff --check

if rg -n 'TODO|FIXME|TBD|작성 예정|추후 작성' \
  main.tex preamble.tex sections references.bib; then
  echo "FAIL: unresolved manuscript marker found." >&2
  exit 1
fi

python3 - <<'PY'
from __future__ import annotations

import re
from pathlib import Path

tex_paths = [Path("main.tex"), Path("preamble.tex"), *sorted(Path("sections").glob("*.tex"))]
text = "\n".join(path.read_text(encoding="utf-8") for path in tex_paths)

controls = [ch for ch in text if ord(ch) < 32 and ch not in "\n\t\r"]
if controls:
    raise SystemExit("FAIL: control character found in manuscript source.")

labels = re.findall(r"\\label\{([^}]+)\}", text)
duplicates = sorted({label for label in labels if labels.count(label) > 1})
if duplicates:
    raise SystemExit(f"FAIL: duplicate labels: {', '.join(duplicates)}")

bib = Path("references.bib").read_text(encoding="utf-8")
bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
cite_groups = re.findall(r"\\cite[pt]?\*?(?:\[[^]]*\])?(?:\[[^]]*\])?\{([^}]+)\}", text)
cite_keys = {key.strip() for group in cite_groups for key in group.split(",")}
missing = sorted(cite_keys - bib_keys)
if missing:
    raise SystemExit(f"FAIL: missing bibliography keys: {', '.join(missing)}")

print(f"PASS: {len(tex_paths)} TeX files, {len(labels)} unique labels, {len(cite_keys)} cited sources")
PY

if [[ -f main.log ]]; then
  if rg -n 'LaTeX Warning: (Reference|Citation).*undefined|There were undefined references|multiply defined|Overfull \\hbox' main.log; then
    echo "FAIL: LaTeX diagnostic warning found in main.log." >&2
    exit 1
  fi
fi

echo "PASS: manuscript validation"
