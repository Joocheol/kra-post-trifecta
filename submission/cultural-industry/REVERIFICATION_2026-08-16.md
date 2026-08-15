# 문화산업연구 원고 재검산 기록 — 2026-08-16

## 목적과 원칙

이 문서는 제출용 원고의 핵심 숫자를 나중에 다시 검산할 수 있도록 계산 과정,
입력 데이터 스냅샷, 난수시드와 반복 수를 함께 고정한다. 주분석 함수의 결과를
그대로 복사하지 않고 `analysis/cultural_industry_reverification.py`에서 역배당 정규화,
삼쌍승 주변화, Harville, TV를 별도로 다시 구현해 계산했다.

원시 KRA JSON gzip 파일은 이 저장소에 재배포되지 않으므로 이번 재검산의 최하단 입력은
버전관리된 `KRA/parsed/` parquet이다. 사용한 모든 parquet의 SHA-256은
`outputs/cultural_industry_reverification_input_files.csv`에 기록했다.

원시자료 자체는 저자의 Dropbox `/kra-analysis/data/raw_collected_v3_15w/` 아래에 연도·월별
`*.json.gz`로 보관되어 있음을 다시 확인했다. 예를 들어
`2025-12-13_2_03.json.gz` 원문에는 쌍승식 매출총액 195,364,900원,
삼복승식 매출총액 839,141,700원, 삼쌍승식 매출총액 411,991,600원이 실제로 들어 있다.
따라서 매출총액은 원자료에 존재하는 관측변수라는 점을 원문 수준에서 재확인했다.
다만 현재 GitHub의 versioned `KRA/parsed/market_status`는 `turnover_won` 열을 포함하지 않는
이전 파싱 스냅샷이다. 아래 turnover-matched 검산은 이 열이 있는 파싱 스냅샷에서 다시
실행하도록 코드만 고정했으며, 현재 실행에서는 `available=False`로 명시적으로 남긴다.

## 1. 표본 재확인

- 분석기간 내 이용 가능한 경주: **19,301개**
- 삼쌍승 9,999.9 표시상한 포함 경주: **15,963개**
- 표시상한 없는 공통 점표본: **3,338개**
- 네 목표 승식의 clean race_id 집합은 서로 완전히 동일함을 다시 확인했다.

중요: 과거 19,284개/3,321개 표본은 2018-07-01의 17개 경주를 제외했기 때문에 나온 값이다.
재검산에서는 9,999.9 **그 자체만** 검열 표시값으로 취급하고, 역사자료에 실제로 존재하는
9,999.9 초과 게시배당은 점관측으로 남긴다. 따라서 현재 원고의 19,301개/3,338개가
버전관리된 parsed data와 현재의 표시상한 정의에 일치한다.

2018-07-01 세부 확인은 `outputs/cultural_industry_2018_overcap_check.csv`에 남겼다.
해당 17개 경주에서 9,999.9와 정확히 같은 값은 0건이지만, 삼복승에는 9,999.9 초과 41개
조합(최대 21,834.2), 삼쌍승에는 3,514개 조합(최대 235,070)이 존재한다. 이 값들을
표시상한으로 잘못 취급하면 17개 경주가 통째로 빠지는 이전 표본오류가 재발한다.

## 2. 핵심 TV의 독립 재계산

TV는 경주별로 `0.5 * sum(abs(observed - predicted))`로 계산했다. 삼쌍승 가격은
각 경주에서 `1/odds`를 합 1로 정규화한 뒤 목표 사건별로 합산했다. Harville은 단승식의
정규화 역배당 확률을 이용해 `(i,j,k)` 순서확률을
`p_i * p_j/(1-p_i) * p_k/(1-p_i-p_j)`로 별도 재구성했다.

|승식|모형|경주수|TV 중앙값|경주 bootstrap 95% CI|경주일 군집 bootstrap 95% CI|
|---|---:|---:|---:|---:|---:|
|쌍승|harville|3,338|0.111965|[0.111033, 0.112973]|[0.110956, 0.113123]|
|쌍승|main|3,338|0.055502|[0.055022, 0.056061]|[0.054942, 0.056174]|
|복승|harville|3,338|0.109101|[0.108140, 0.110350]|[0.108121, 0.110357]|
|복승|main|3,338|0.063653|[0.062964, 0.064617]|[0.062868, 0.064700]|
|삼복승|harville|3,338|0.152002|[0.150130, 0.153319]|[0.150113, 0.153375]|
|삼복승|main|3,338|0.044383|[0.043885, 0.044867]|[0.043637, 0.044967]|

bootstrap 반복 수는 **4,999회**이며 고정 기준시드는 `20260816`이다.
경주 bootstrap과 경주일 군집 bootstrap을 모두 남겼고, 후자는 같은 날짜의 여러 경주를
한 군집으로 묶어 날짜를 복원추출한다. `tracked_matches=True`가 전 행에서 확인되어야 하며,
이는 독립 재계산 중앙값이 기존 `outputs/main_panel_a_summary.csv`와 1e-10 이내로 일치함을 뜻한다.

## 3. 매출액과 유한 풀 기계적 기준

원자료에는 경주×승식별 매출총액이 존재한다. `market_status.turnover_won`이 있는 파싱
스냅샷에서는 실제 경주×승식 매출액을 사용하도록 검산 코드를 작성해 두었다. 단, 매출액을
100원으로 나눈 수를 독립 베팅 의사결정 수로 해석하지 않는다. 기계적 null에서만 100원
단위를 독립 다항추출로 가정한다. 이 null의 Monte Carlo 반복 수는 경주당 **256회**,
기준시드는 `20260816`이다.

현재 GitHub 파싱 스냅샷에는 `turnover_won` 열이 없으므로
`outputs/cultural_industry_turnover_limit_summary.csv`에 `available=False`가 기록되었고,
turnover-matched 결과를 논문 수치로 사용하지 않았다. 향후 Dropbox 원자료를 revised parser로
재파싱하거나 turnover 열만 별도 compact CSV로 추출하면 같은 코드가 자동으로 이 부분까지
재계산한다. 원자료 전체를 공개 GitHub에 복제하는 대신, 재파싱 시에는 경주×승식별 turnover
compact data와 그 원천파일 hash/provenance만 GitHub에 동결하는 방식을 권장한다.

## 4. 재현 명령

```bash
python -m pip install -r requirements.txt -c constraints-behavioral.txt
python -m unittest tests.test_cultural_industry_reverification -v
python -m analysis.cultural_industry_reverification
```

## 5. 고정 산출물

- `outputs/cultural_industry_reverification_summary.csv`: TV 중앙값과 두 bootstrap CI
- `outputs/cultural_industry_reverification_per_race.csv`: 경주별 TV 재계산값
- `outputs/cultural_industry_reverification_sample.csv`: 표본 수와 상한 수 재감사
- `outputs/cultural_industry_2018_overcap_check.csv`: 2018-07-01의 9,999.9/초과배당 진단
- `outputs/cultural_industry_turnover_limit_summary.csv`: turnover 열 가용성 및 10만원 상한 진단
- `outputs/cultural_industry_turnover_null_summary.csv`: 실제 매출액 일치 기계적 null 요약(현재 snapshot에서는 빈 결과)
- `outputs/cultural_industry_turnover_null_per_race.csv`: 경주별 null 결과(현재 snapshot에서는 빈 결과)
- `outputs/cultural_industry_reverification_input_files.csv`: 사용 parquet 전부의 SHA-256
- `outputs/cultural_industry_reverification_manifest.json`: 코드·환경·파라미터와 산출물 hash

입력 parquet 파일 수: **637개**.
