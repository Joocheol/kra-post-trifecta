# KRA Post-Trifecta Dataset

Parsed KRA betting-market data for the post-trifecta era, with the scripts
used to build it and a static data browser.

## Data Scope

- Race-date range: `2016-06-10` through `2025-12-31`
- Excluded years: `2020`, `2021` (COVID-19 period)
- Excluded date: `2018-07-01` — TODO: 제외 사유를 여기에 적으세요
- Cutover rationale: `trifecta` betting begins `2016-06-10`, so this dataset
  starts at the post-trifecta market era.
- Parsed parquet dataset: `KRA/parsed/`
- Raw JSON source: not included in this repository (see Data Source below)

Note: excluded dates are omitted from the analysis scope, but the underlying
records may still appear in the static browser under `docs/`.

## Included Parsed Partitions

- `kra_2016_post`
- `kra_2017`, `kra_2018`, `kra_2019`
- `kra_2022`, `kra_2023`, `kra_2024`, `kra_2025`

Markets per partition: `win`, `place`, `quinella`, `quinella_place`,
`exacta`, `trifecta`, `trio`, plus `market_status` and `valid_horses`.
Partitioned by year and month.

Race counts for this subset: 19,301 races.

## Data Source and License

Source: Korea Racing Authority (한국마사회), race.kra.co.kr public data archive.
Collected by the author via automated retrieval for research purposes.

Redistributed here for academic and educational use only. Rights in the
underlying data remain with the Korea Racing Authority. Equivalent data is
published by KRA as open APIs at data.go.kr.

## Python Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Parsed files are parquet datasets intended to be read with `pyarrow`.

## Scripts

- `scripts/kra_reparse_raw.py` — rebuilds `KRA/parsed/` from the raw JSON
  gzip archive. Documents the raw-to-parsed transformation.
- `scripts/build_site_data.py` — regenerates site metadata
- `scripts/build_static_odds_data.py` — regenerates static odds files
- `scripts/serve_data_browser.py` — local preview server

## Data Browser

The static browser lives in `docs/` and is served via GitHub Pages.

Regenerate:

```bash
python scripts/build_site_data.py
python scripts/build_static_odds_data.py
```

Preview locally at `http://localhost:8000`:

```bash
python scripts/serve_data_browser.py
```
