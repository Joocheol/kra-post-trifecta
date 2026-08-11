# 행동모형 분석 구현

## 현재 범위

이 모듈은 주분석과 별도로 다음 세 단계를 구현한다.

1. 2016--2019년과 2022--2025년을 fold로 둔 leave-one-year-out 조건부
   1--3위 확률 추정·검증
2. 단승 가격에서만 추정한 확률가중 함수를 쌍승·삼쌍승의 전체 비검열
   가격벡터로 이동시킨 M-U/M-R/M-S2/M-S3 비교
3. 복승·삼복승에서 순서 없는 사건을 한 번 평가한 M-R과, 대응하는 두 개 또는
   여섯 개 순서형 청구권의 M-S2/M-S3 예측가격을 합한 외부검증
4. 각 검증연도보다 엄격히 앞선 연도만 훈련에 사용하는 expanding-window
   시간순 민감도 분석

실현 1--3위는 `valid_horses.arrival_order`에서 읽는다. 분석기간의 19,284개
경주는 모두 서로 다른 1--3위와 유효 출전마의 대응을 통과한다. 가격모형은
게시 상한을 점값으로 취급하지 않는다. 따라서 각 승식은 해당 승식의 전체
가격벡터가 비검열인 경주만 사용한다. 동결 표본수는 분석 manifest에 기록한다.

## 조건부 순위확률

단승 풀의 정규화 역배당률을 실현 우승 여부에 단조 isotonic calibration하여
객관 우승확률을 만든다. 검증연도의 착순은 이 함수 추정에 들어가지 않는다.

- `harville`: 보정된 우승확률을 남은 말 사이에서 그대로 재정규화한다.
- `stage_temperature`: 2·3위 단계에서 남은 말의 우승확률을
  `p ** alpha_s(n_s)`으로 변환한 뒤 재정규화한다. 여기서 `n_s`는 해당
  단계의 남은 선택집합 크기이며, `alpha_s(n_s)`은 훈련연도의 조건부
  선택우도로만 추정한다.

전체 19,284개 경주의 표본 밖 검증에서 단계조정 모형은 Harville 대비 2위
로그손실을 약 1.5%, 3위 로그손실을 약 3.7% 낮춘다. 남은 선택집합이
10마리일 때 추정치는
2위 약 0.73--0.75, 3위 약 0.56--0.58로 여덟 fold에서 안정적이다.

## 가격모형과 식별

훈련연도의 객관 우승확률과 단승 가격지분 사이에서 다음 함수를 고정한다.

- `isotonic_clip`: 단조 비모수 함수. 훈련확률의 1--99 백분위 밖에서는
  끝점으로 고정하므로 꼬리 민감도 명세로 표시한다.
- `prelec`: 사전 지정한 Prelec 함수형 꼬리 외삽
- `power`: power 함수형. 이 경우 `w(ab)=w(a)w(b)`이므로 축약·순차 모형이
  이론적으로 같아지는 음성대조군이다.

가격모형은 다음과 같다.

- `M-U`: 단승에서 비모수적으로 회수한 효용표현. 제한 없는 경우 M-R과
  관측적으로 동등하므로 수치도 M-R과 같게 보고한다.
- `M-R`: 결합확률에 가중함수를 한 번 적용한다.
- `M-S2`: 쌍승은 1위와 조건부 2위, 삼쌍승은 1위와 조건부 2·3위 결합사건을
  분리한다.
- `M-S3`: 삼쌍승의 1위, 조건부 2위, 조건부 3위를 모두 분리한다.

복승에서는 M-S2의 두 순서형 쌍승 청구권 점수를 합하고, 삼복승에서는 M-S2와
M-S3의 여섯 순서형 삼쌍승 청구권 점수를 각각 합한다. 경주 안에서 이 합을
정규화하는 것은 먼저 순서형 전체 가격벡터를 정규화한 뒤 대응 순서를 합하는
것과 같다. M-R은 순서 없는 결합사건의 확률합에 가중함수를 한 번 적용한다.
이 과정에서 복승·삼복승에 자의적인 단계 순서를 부여하지 않는다.

power 음성대조는 하나의 순서형 사건을 곱으로 분해할 때에만 M-R과 M-S가
일치한다. 무순서 사건은 서로 배반인 순서 사건의 합집합이므로 일반적으로
`w(sum p) != sum w(p)`이며, power 동등성 검사를 무순서 외부검증에 적용하지
않는다.

주 비교는 경주 안에서 정규화한 전체 가격벡터의 TV, MAE, 로그 RMSE와 JS이다.
이 지표들은 모든 예측가격에 곱하는 양의 수준상수에 불변하므로, 본문의
축약 대 순차평가 모형순위는 수준상수 추정에 의존하지 않는다. 원시 `1/D`
MAE는 생성 산출물의 보조진단이며, 여기에만 쓰이는 수준상수는 검증연도의 실제
overround를 사용하지 않고 각 fold의 다른 연도 비검열 경주에서 추정해
검증연도에서 고정한다. 쌍승·삼쌍승의 추정 상수는 모두 약 1.36986이다.

네 그림의 내부 제목·축·범례는 국내 경제학 논문의 압축적 도표 관행과 PDF
폰트 임베딩의 일관성을 위해 영문으로 유지한다. 본문 설명과 그림 캡션은
한국어로 제공한다.

## 1차 결과

선호 명세인 단계조정 순위확률과 Prelec 꼬리모형에서 TV 중앙값은 다음과 같다.

| 시장 | M-R | M-S2 | M-S3 |
|---|---:|---:|---:|
| 쌍승 | 0.2534 | 0.1264 | 해당 없음 |
| 삼쌍승 | 0.3178 | 0.2075 | 0.1578 |
| 복승 | 0.1833 | 0.1300 | 해당 없음 |
| 삼복승 | 0.2775 | 0.2204 | 0.1545 |

경주별 paired TV 개선폭의 중앙값과 999회 경주일 군집 부트스트랩 95% 구간은 다음과
같다.

| 비교 | 개선폭 중앙값 | 95% 구간 | 양의 연도 수 |
|---|---:|---:|---:|
| 쌍승 M-R → M-S2 | 0.1230 | [0.1220, 0.1243] | 8/8 |
| 삼쌍승 M-R → M-S2 | 0.1069 | [0.1049, 0.1082] | 8/8 |
| 삼쌍승 M-R → M-S3 | 0.1554 | [0.1533, 0.1577] | 8/8 |
| 삼쌍승 M-S2 → M-S3 | 0.0498 | [0.0485, 0.0511] | 8/8 |
| 복승 M-R → M-S2 | 0.0557 | [0.0544, 0.0569] | 8/8 |
| 삼복승 M-R → M-S2 | 0.0543 | [0.0538, 0.0548] | 8/8 |
| 삼복승 M-R → M-S3 | 0.1213 | [0.1201, 0.1228] | 8/8 |
| 삼복승 M-S2 → M-S3 | 0.0663 | [0.0654, 0.0675] | 8/8 |

다만 M-R의 모든 가중함수 인수가 단승 공통지지에 들어오는 조합 비율은 쌍승
69.5%, 삼쌍승 38.1%, 복승 81.4%, 삼복승 63.9%다. M-S2는 각각
98.0%, 98.0%, 96.1%, 74.7%이고 M-S3은 삼쌍승 98.9%, 삼복승
96.5%다. 따라서 M-R의 절대 성능은 꼬리 외삽에 더 민감하며, 강한 결론은
함수형 하나가 아니라 isotonic/Prelec 방향 일치와 power 음성대조를 함께
보고해야 한다.

## 시간순 검증

leave-one-year-out 결과가 미래 연도 훈련자료에 의존하는지 확인하기 위해,
2017--2025년의 각 검증연도에서 이전 연도만 누적해 동일 모형을 다시 추정한다.
2016년은 이전 훈련연도가 없어 제외한다. 시간순 표본은 착순확률 17,745개 경주,
쌍승 17,324개, 삼쌍승 3,130개, 복승 17,736개와 삼복승 14,840개 경주다.

단계조정 모형의 Harville 대비 로그손실 감소율은 2위 1.47%, 3위 3.65%다.
선호 명세의 TV 중앙값은 쌍승 M-R/M-S2가 0.2664/0.1326, 삼쌍승
M-R/M-S2/M-S3이 0.3258/0.2143/0.1596이다. 복승은
0.1955/0.1341, 삼복승은 0.2906/0.2317/0.1599이며, 여덟 TV
대비는 모두 7개 검증연도에서 같은 방향이다.

## 재현 명령

```bash
python -m pip install -r requirements.txt -c constraints-behavioral.txt
python -m analysis.behavioral_analysis
python -m analysis.behavioral_figures
python -m analysis.behavioral_report
python -m unittest discover -s tests -v
bash scripts/validate.sh
```

`--max-races`는 개발 전용이다. 최종 동결 산출물에는 사용하지 않는다.

## 산출물

- `outputs/rank_probability_validation.csv`
- `outputs/rank_probability_validation_by_year.csv`
- `outputs/behavioral_model_parameters.csv`
- `outputs/behavioral_model_comparison.csv`
- `outputs/behavioral_model_improvements.csv`
- `outputs/rank_probability_time_forward.csv`
- `outputs/rank_probability_time_forward_by_year.csv`
- `outputs/behavioral_time_forward_parameters.csv`
- `outputs/behavioral_time_forward_comparison.csv`
- `outputs/behavioral_time_forward_improvements.csv`
- `outputs/behavioral_analysis_manifest.json`
- `figures/calibration-rank-probabilities.pdf`
- `figures/model-comparison-behavioral.pdf`
- `figures/model-comparison-unordered.pdf`
- `figures/model-comparison-support.pdf`
- `tables/behavioral_rank_validation.tex`
- `tables/behavioral_model_comparison.tex`
- `tables/behavioral_model_improvements.tex`
- `tables/behavioral_time_forward.tex`

180MB 내외의 `outputs/behavioral_model_metrics_by_race.csv`는 결정론적 재생성
산출물이므로 Git에서 제외한다.
