# 원시자료 provenance 및 매출총액 재확인 — 2026-08-16

## 보관 위치

원시 KRA 수집자료는 저자의 Dropbox 아래에 연도·월·경주별 gzip JSON으로 보관되어 있음을 다시 확인했다.

- 루트: `/kra-analysis/data/raw_collected_v3_15w/`
- 연도 폴더: `kra_2016`, `kra_2017`, `kra_2018`, `kra_2019`, `kra_2022`, `kra_2023`, `kra_2024`, `kra_2025`
- 형식: `raw_archive/<year>/<year-month>/<race_id>.json.gz`

원시자료 전체는 크기가 크고 재배포 범위를 별도로 관리해야 하므로 공개 GitHub 저장소에 복제하지 않는다. 재검산에 실제로 사용한 공개 저장소의 파싱자료 637개 parquet은 별도 SHA-256 목록(`outputs/cultural_industry_reverification_input_files.csv`)으로 고정했다.

## 매출총액 원문 확인

표본 원문으로 다음 파일을 직접 열어 확인했다.

`/kra-analysis/data/raw_collected_v3_15w/kra_2025/raw_archive/2025/2025-12/2025-12-13_2_03.json.gz`

이 원문은 각 승식 페이지의 HTML/text를 보존하며, 적어도 다음 매출총액이 명시되어 있다.

- 쌍승식: **195,364,900원**
- 삼복승식: **839,141,700원**
- 삼쌍승식: **411,991,600원**

따라서 경주×승식별 매출총액은 원자료에서 관측 가능한 변수라는 점을 원문 수준에서 재확인했다.

## 현재 GitHub 파싱 스냅샷의 한계

2026-08-16 독립 재검산에 사용한 공개 GitHub `KRA/parsed/market_status` 스냅샷에는 `turnover_won` 열이 아직 들어 있지 않다. 이 때문에 `analysis/cultural_industry_reverification.py`의 turnover-matched 기계적 null은 현재 실행에서 명시적으로 사용할 수 없음(`available=False`)으로 남겼고, 해당 결과를 논문 수치로 사용하지 않았다.

향후 매출액 기반 검산을 다시 수행할 때의 절차는 다음과 같이 고정한다.

1. Dropbox 원시 `*.json.gz`를 `scripts/kra_reparse_raw.py`로 다시 파싱하여 `market_status.turnover_won`을 보존한다.
2. 원시자료 전체 대신 경주×승식별 compact turnover data와 원천 provenance/hash를 GitHub에 동결한다.
3. `python -m analysis.cultural_industry_reverification`을 다시 실행한다.
4. 매출총액을 100원으로 나눈 값은 기계적 null의 명목 시행 수로만 사용하고, 실제 독립 베팅 의사결정 수로 해석하지 않는다.
5. 경주당 개인 구매상한 100,000원을 이용한 수치는 최소 참여자-경주 수의 하한으로만 해석한다.

기계가 읽을 수 있는 동일 내용은 `outputs/cultural_industry_raw_provenance.json`에 기록해 두었다.
