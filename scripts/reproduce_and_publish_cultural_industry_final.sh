#!/usr/bin/env bash
set -euo pipefail

# Complete the turnover-dependent Table 4 from the locally synced Dropbox raw
# archive, re-run the unified manuscript calculation, freeze the compact turnover
# data and final outputs, and push them to the reverification branch.

BRANCH="agent/cultural-industry-reverification"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean. Commit/stash local changes first." >&2
  exit 2
fi

git fetch origin "$BRANCH"
current="$(git branch --show-current)"
if [[ "$current" != "$BRANCH" ]]; then
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git switch "$BRANCH"
  else
    git switch --track -c "$BRANCH" "origin/$BRANCH"
  fi
fi
git pull --ff-only origin "$BRANCH"

bash scripts/reproduce_cultural_industry_final.sh

python - <<'PY'
from pathlib import Path
import pandas as pd
status = Path('outputs/final_19301/table4_status.csv')
if not status.exists():
    raise SystemExit('ERROR: Table 4 status file not generated')
frame = pd.read_csv(status)
if frame.empty or not bool(frame.iloc[0]['available']):
    raise SystemExit(
        'ERROR: Table 4 is still unavailable. Confirm that the Dropbox raw archive '
        'exists at ~/Library/CloudStorage/Dropbox/kra-analysis/data/raw_collected_v3_15w '
        'or set KRA_RAW_ROOT explicitly.'
    )
print('PASS: Table 4 turnover data and manuscript outputs are complete.')
PY

git add \
  data/turnover_by_race_market.csv.gz \
  data/turnover_by_race_market.manifest.json \
  outputs/final_19301

if git diff --cached --quiet; then
  echo "No new turnover/final-output changes to commit."
else
  git commit -m "Freeze turnover-based final cultural-industry tables"
fi

git push origin "$BRANCH"

echo "PASS: complete Tables 1-5 reproduction is published to $BRANCH"
