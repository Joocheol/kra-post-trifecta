# KRA Post-Trifecta: 데이터와 논문

이 저장소는 삼쌍승식 도입 이후 한국 경마 베팅시장 데이터와 이를 이용한
한글 LaTeX 논문을 함께 관리합니다. 데이터·브라우저의 기존 기능은 유지하며,
논문은 삼쌍승식의 순위 상태가격을 주변화하여 단승·쌍승·복승·삼복승 가격을
얼마나 재구성할 수 있는지를 주분석으로 삼습니다. 별도의 행동모형 확장에서는
단승식에서 추정한 위험선호·확률가중 함수와 복합복권의 축약·3단계 순차평가를
다른 승식의 전체 가격벡터에서 비교합니다.

## Paper

- 가제: **경마 베팅시장의 정보 집계: 삼쌍승식 상태가격의 교차풀 정합성과 3단계 행동모형**
- 원고: `main.tex`, `preamble.tex`, `sections/`
- 연구 설계: `RESEARCH_PLAN.md`
- 집필 원칙: `WRITING_GUIDE.md`
- Claude 검토 규칙: `CLAUDE.md`

로컬 빌드 환경에 XeLaTeX, ko.TeX, BibTeX이 설치되어 있다면 다음 두 명령으로
검증하고 PDF를 만듭니다.

```bash
bash scripts/validate.sh
bash scripts/build.sh
```

PR에서는 `paper-ci.yml`이 같은 검증을 수행하고 `paper-pdf` artifact를 남깁니다.
`ai-review` 라벨은 별도의 읽기 전용 Claude 전수 검토를 한 번 실행합니다. 이
검토는 PDF의 한글 추출을 확인하고, 모든 쪽을 PNG로 변환한 뒤 전 페이지를
읽습니다. Claude는 파일을 수정하거나 병합하지 않습니다.

## Data Scope

- Race-date range: `2016-06-10` through `2025-12-31`
- Excluded years: `2020`, `2021` (COVID-19 period)
- Excluded date: `2018-07-01` — 9999 초과 데이터 존재
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

Candidate race count for this date-restricted subset: 19,301 races. This is the
pre-filter count; the paper's main-analysis inclusion conditions are applied later.

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
- `analysis/data_audit.py` — audits race/market support, keys, odds validity and
  capped odds; freezes the complete, clean and interval-analysis samples.
- `scripts/build_site_data.py` — regenerates site metadata
- `scripts/build_static_odds_data.py` — regenerates static odds files
- `scripts/serve_data_browser.py` — local preview server
- `scripts/validate.sh` — 논문 소스와 저장소의 정적 검증
- `scripts/build.sh` — XeLaTeX/BibTeX 논문 빌드

## Freeze the analysis sample

Run the data audit before computing reconstruction metrics:

```bash
python -m analysis.data_audit --strict
```

The audit writes race-level evidence to `outputs/data_quality.csv`, target-specific
sample membership to `outputs/analysis_sample.csv`, the sequential sample counts to
`outputs/sample_flow.csv`, and a compact interpretation to
`outputs/data_audit_summary.md`. The manuscript-ready audit table is generated at
`tables/data_quality_summary.tex`. Capped trifecta odds are not silently treated as
point observations: the clean-sample point estimates (Panel A) and full-sample
partial-identification bounds (Panel B) are co-primary results.

The two race-level CSVs are deterministic but intentionally not versioned. Their row
counts and embedded SHA-256 hashes are frozen in `outputs/data_audit_manifest.json`.
CI regenerates both CSVs and the manifest, then byte-compares the tracked manifest;
this indirect hash comparison is the freshness check for the untracked CSVs.

## Behavioral-model analysis

Run the leave-one-year-out rank-probability validation and the win-to-exacta/
trifecta price transfer after the sample audit:

```bash
python -m analysis.behavioral_analysis
MPLCONFIGDIR=/tmp/matplotlib python -m analysis.behavioral_figures
```

The implementation contract, identified contrasts and current caveats are recorded
in `BEHAVIORAL_ANALYSIS_IMPLEMENTATION.md`. The race-level behavioral metric file is
regenerated but not versioned; compact summaries and deterministic PDF figures are
tracked and checked by Paper CI.

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
