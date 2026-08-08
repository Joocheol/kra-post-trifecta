# KRA Post-Trifecta Analysis

This repository contains a KRA betting-market data subset for a fresh analysis workflow.

## Data Scope

- Race-date range: `2016-06-10` through `2025-12-31`
- Excluded years: `2020`, `2021` (COVID period)
- Cutover rationale: `trifecta` is available from `2016-06-10`, so this repo starts at the post-trifecta market era.
- Raw source subset: `KRA/raw_collected_v3_15w/`
- Parsed parquet subset: `KRA/parsed/`
- Parsed source: regenerated from the raw JSON gzip files with `scripts/kra_reparse_raw.py`

## Included Parsed Partitions

- `kra_2016_post`
- `kra_2017`, `kra_2018`, `kra_2019`
- `kra_2022`, `kra_2023`, `kra_2024`, `kra_2025`

The current raw and parsed race universes match exactly for this subset:

- Raw races: `19,301`
- Parsed races: `19,301`
- Parsed-only races: `0`
- Raw-only races: `0`

## Python Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data Browser

This repo includes a static data browser in `docs/`.

Regenerate the site metadata:

```bash
python scripts/build_site_data.py
python scripts/build_static_odds_data.py
```

Preview locally:

```bash
python scripts/serve_data_browser.py
```

Then open `http://localhost:8000`.

The date-level odds table uses the local API server above. The static files in `docs/`
can also be hosted by GitHub Pages after running `build_static_odds_data.py`.

## Notes

Raw files are stored as compressed JSON files. Parsed files are parquet datasets intended to be read with `pyarrow`.
