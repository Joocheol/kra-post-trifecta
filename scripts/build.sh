#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
KEEP_AUX="${PAPER_KEEP_AUX:-0}"

cd "$ROOT_DIR"

clean_aux_files() {
  find "$ROOT_DIR" \
    -type f \
    ! -path "$ROOT_DIR/.git/*" \
    \( -name '*.aux' \
       -o -name '*.bbl' \
       -o -name '*.blg' \
       -o -name '*.fdb_latexmk' \
       -o -name '*.fls' \
       -o -name '*.log' \
       -o -name '*.out' \
       -o -name '*.run.xml' \
       -o -name '*.synctex.gz' \
       -o -name '*.toc' \
       -o -name '*.xdv' \) \
    -delete
}

clean_aux_files
echo "[kra-post-trifecta] Building Korean paper with XeLaTeX and BibTeX..."

if latexmk \
  -xelatex \
  -bibtex \
  -synctex=1 \
  -interaction=nonstopmode \
  -file-line-error \
  -halt-on-error \
  main.tex; then
  test -s main.pdf
  if [[ "$KEEP_AUX" == "1" ]]; then
    echo "[kra-post-trifecta] Created main.pdf and retained diagnostics."
  else
    clean_aux_files
    echo "[kra-post-trifecta] Created main.pdf and removed auxiliary files."
  fi
else
  echo "[kra-post-trifecta] Build failed; diagnostic files were retained." >&2
  exit 1
fi
