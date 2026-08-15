#!/usr/bin/env bash
set -euo pipefail

# One-command clean-checkout reproduction for the Cultural Industry manuscript.
# Optional: export KRA_RAW_ROOT=/path/to/raw_collected_v3_15w to build the compact
# turnover extract needed for Table 4. If the compact file is already versioned,
# KRA_RAW_ROOT is not needed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pip install -r requirements.txt -c constraints-behavioral.txt
python -m analysis.data_audit --strict

TURNOVER_FILE="data/turnover_by_race_market.csv.gz"
if [[ ! -s "$TURNOVER_FILE" && -n "${KRA_RAW_ROOT:-}" ]]; then
  python -m analysis.build_turnover_dataset \
    --raw-root "$KRA_RAW_ROOT" \
    --output "$TURNOVER_FILE" \
    --manifest data/turnover_by_race_market.manifest.json
fi

if [[ -s "$TURNOVER_FILE" ]]; then
  python -m analysis.cultural_industry_final_recompute
else
  echo "WARNING: $TURNOVER_FILE is absent; Tables 1-3 and 5 plus all diagnostics will be reproduced, but Table 4 will be marked unavailable." >&2
  echo "Set KRA_RAW_ROOT to the archived Dropbox raw root, or supply the versioned compact turnover file, for the complete Tables 1-5 run." >&2
  python -m analysis.cultural_industry_final_recompute --allow-missing-turnover
fi

python submission/cultural-industry/validate.py

echo "PASS: Cultural Industry final reproduction completed"
