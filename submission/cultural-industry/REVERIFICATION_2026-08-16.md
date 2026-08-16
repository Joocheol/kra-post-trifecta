# 문화산업연구 원고 재검산 기록 — 2026-08-16

## 목적과 원칙

이 문서는 제출용 원고의 핵심 숫자를 나중에 다시 검산할 수 있도록 계산 과정, 입력 데이터 스냅샷, 난수시드와 반복 수를 함께 고정한다. 과거 19,284개/3,321개 표본은 2018-07-01의 17개 경주를 제외했기 때문에 나온 값이며, 최종 사양에서는 정확히 9,999.9인 게시값만 표시상한으로 취급한다.

## 1. 최종 표본

- 분석기간 내 이용 가능한 경주: **19,301개**
- 삼쌍승 9,999.9 표시상한 포함 경주: **15,963개**
- 표시상한 없는 공통 점표본: **3,338개**
- 점표본 경주일: **979일**
- 2018-07-01의 17개 경주는 다섯 승식 모두 exact 9,999.9 = 0건이므로 최종 표본에 포함한다.

전체 표본의 표시상한 경주 수는 단승 0, 쌍승 581, 복승 15, 삼복승 3,434, 삼쌍승 15,963이다.

## 2. 최종 추론 규칙

원고에 들어가는 모든 최종 불확실성 구간은 다음 규칙으로 통일한다.

- 재표집 단위: **경주일 군집**
- 반복 수: **99,999회**
- seed: `20260816`
- 구간: percentile 95% CI

초기 독립 재검산에서 사용했던 4,999회 경주/경주일 bootstrap은 계산 경로를 점검하기 위한 중간 검증이었다. 원고와 `outputs/final_19301/`의 최종 숫자는 99,999회 경주일 군집 결과를 사용한다.

## 3. Panel A 핵심 TV

|승식|모형|경주수|TV 중앙값|경주일 군집 95% CI|
|---|---:|---:|---:|---:|
|쌍승|Harville|3,338|0.111965|[0.110954, 0.113122]|
|쌍승|재구성|3,338|0.055502|[0.054948, 0.056142]|
|복승|Harville|3,338|0.109101|[0.108107, 0.110421]|
|복승|재구성|3,338|0.063653|[0.062895, 0.064697]|
|삼복승|Harville|3,338|0.152002|[0.150081, 0.153364]|
|삼복승|재구성|3,338|0.044383|[0.043676, 0.044981]|

세 복합승식 모두에서 삼쌍승 주변가격이 Harville보다 실제 시장가격에 가깝다.

## 4. 실제 매출액과 Table 4

경주×승식 매출총액은 KRA 상세 성적표에서 추출해 compact 파일로 동결했다.

- 파일: `data/turnover_by_race_market.csv.gz`
- 열: `race_id`, `race_date`, `market`, `turnover_won`
- 범위: 점표본 3,338경주 × 쌍승·복승·삼복승·삼쌍승 = **13,352행**
- SHA-256: `3a5a899933479a432ef75cbb8699c8128f7ecd325cedd5f1e58cede3f4247be6`
- Dropbox 원자료 대조 경주: `2025-12-13_2_03`
- 대조값: 복승 763,893,500원, 쌍승 195,364,900원, 삼복승 839,141,700원, 삼쌍승 411,991,600원

Table 4는 두 풀이 같은 잠재가격을 공유한다는 귀무가정 아래, 각 풀의 총매출과 유효 베팅단위 u로부터 두 독립 다항 풀의 정상근사 잡음 TV를 계산한다. u는 실제 평균 베팅액의 추정치가 아니라 민감도 분석용 유효 독립 단위다.

|승식|관측 TV 중앙값|u=1만원 잡음 TV (관측 대비)|u=5만원|u=10만원|손익분기 u 중앙값|
|---|---:|---:|---:|---:|---:|
|쌍승|0.0555016|0.0230109 (41.5%)|0.0514539 (92.7%)|0.0727669 (131.1%)|56,368원|
|복승|0.0636531|0.0127266 (20.0%)|0.0284575 (44.7%)|0.0402450 (63.2%)|248,283원|
|삼복승|0.0443832|0.0191065 (43.0%)|0.0427234 (96.3%)|0.0604200 (136.1%)|53,826원|

비율은 미반올림 원값에서 계산한 뒤 마지막 단계에서만 소수 첫째 자리로 반올림한다. 과거 3,321경주 표본에서 얻은 Table 4 숫자는 최종 원고용 값이 아니며 폐기한다.

전체 원값과 CI는 `outputs/final_19301/table4_turnover_noise.csv`, 경주별 계산은 `outputs/final_19301/table4_turnover_noise_per_race.csv`, 계산 정의는 `outputs/final_19301/TABLE4_METHOD.md`에 보존한다.

## 5. 재현 명령

compact turnover 파일이 체크인된 현재 상태에서는 다음 명령으로 최종 원고용 산출물을 다시 계산할 수 있다.

```bash
python -m pip install -r requirements.txt -c constraints-behavioral.txt
python -m unittest tests.test_cultural_industry_reverification -v
python -m analysis.cultural_industry_final_recompute
```

Dropbox 원자료에서 compact turnover 파일 자체를 다시 만들 때에는 다음 경로를 사용한다.

```bash
bash scripts/reproduce_and_publish_cultural_industry_final.sh
```

## 6. 최종 고정 산출물

- `data/turnover_by_race_market.csv.gz`: 경주×승식 매출총액 compact data
- `data/turnover_by_race_market.manifest.json`: 출처·대조·hash
- `outputs/final_19301/table3_panel_a.csv`: 점표본 TV와 CI
- `outputs/final_19301/table3_panel_b.csv`: 전체표본 부분식별 결과
- `outputs/final_19301/table4_turnover_noise.csv`: 실제 매출액 기반 Table 4
- `outputs/final_19301/table4_turnover_noise_per_race.csv`: 경주별 Table 4 계산
- `outputs/final_19301/table5_logloss.csv`: 로그손실 결과
- `outputs/final_19301/MANUSCRIPT_REPLACEMENT_TABLES.md`: 원고 교체용 최종 표
- `outputs/final_19301/manifest.json`: 최종 산출물과 환경 기록

현재 최종 동결 사양은 **19,301 / 3,338 / 979일 / 경주일 군집 bootstrap 99,999회 / seed 20260816**이다.
