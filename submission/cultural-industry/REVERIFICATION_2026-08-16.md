# 문화산업연구 원고 재검산 기록 — 2026-08-16

## 목적과 원칙

이 문서는 제출용 원고의 핵심 숫자를 나중에 다시 검산할 수 있도록 계산 과정,
입력 데이터 스냅샷, 난수시드와 반복 수를 함께 고정한다. 주분석 함수의 결과를
그대로 복사하지 않고 `analysis/cultural_industry_reverification.py`에서 역배당 정규화,
삼쌍승 주변화, Harville, TV를 별도로 다시 구현해 계산했다.

원시 KRA JSON gzip 파일은 이 저장소에 재배포되지 않으므로 이번 재검산의 최하단 입력은
버전관리된 `KRA/parsed/` parquet이다. 사용한 모든 parquet의 SHA-256은
`outputs/cultural_industry_reverification_input_files.csv`에 기록했다.

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

`market_status.turnover_won`이 있으면 실제 경주×승식 매출액을 사용한다. 단, 매출액을 100원으로
나눈 수를 독립 베팅 의사결정 수로 해석하지 않는다. 기계적 null에서만 100원 단위를 독립 다항
추출로 가정한다. 이 null의 Monte Carlo 반복 수는 경주당
**256회**, 기준시드는 `20260816`이다.

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
- `outputs/cultural_industry_turnover_limit_summary.csv`: 총매출과 10만원 상한의 하한 진단
- `outputs/cultural_industry_turnover_null_summary.csv`: 실제 매출액 일치 기계적 null 요약
- `outputs/cultural_industry_turnover_null_per_race.csv`: 경주별 null 결과
- `outputs/cultural_industry_reverification_input_files.csv`: 사용 parquet 전부의 SHA-256
- `outputs/cultural_industry_reverification_manifest.json`: 코드·환경·파라미터와 산출물 hash

입력 parquet 파일 수: **637개**.
