# 행동모형 분석 구현

## 현재 범위

이 모듈은 주분석과 별도로 다음 두 단계를 구현한다.

1. 2016--2019년과 2022--2025년을 fold로 둔 leave-one-year-out 조건부
   1--3위 확률 추정·검증
2. 단승 가격에서만 추정한 확률가중 함수를 쌍승·삼쌍승의 전체 비검열
   가격벡터로 이동시킨 M-U/M-R/M-S2/M-S3 비교

실현 1--3위는 `valid_horses.arrival_order`에서 읽는다. 분석기간의 19,284개
경주는 모두 서로 다른 1--3위와 유효 출전마의 대응을 통과한다. 가격모형은
게시 상한을 점값으로 취급하지 않는다. 따라서 쌍승은 18,703개, 삼쌍승은
3,321개 비검열 경주를 사용한다.

복승·삼복승으로의 무순서 외부 검증은 다음 PR 범위다. 현재 결과는 순서형
쌍승·삼쌍승의 풀 밖 형태 이동성에 관한 것이다.

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

주 비교는 경주 안에서 정규화한 전체 가격벡터의 TV, MAE, 로그 RMSE와 JS이다.
원시 `1/D` 수준은 검증연도의 실제 overround를 사용하지 않는다. 각 fold의
다른 연도 비검열 경주에서 승식별 수준상수 하나를 추정하고 검증연도에서
고정한다. 쌍승·삼쌍승의 추정 상수는 모두 약 1.36986이다.

## 1차 결과

선호 명세인 단계조정 순위확률과 Prelec 꼬리모형에서 TV 중앙값은 다음과 같다.

| 시장 | M-R (= M-U) | M-S2 | M-S3 |
|---|---:|---:|---:|
| 쌍승 | 0.2534 | 0.1264 | 해당 없음 |
| 삼쌍승 | 0.3178 | 0.2075 | 0.1578 |

경주별 paired TV 개선폭의 중앙값과 999회 경주 부트스트랩 95% 구간은 다음과
같다.

| 비교 | 개선폭 중앙값 | 95% 구간 | 양의 연도 수 |
|---|---:|---:|---:|
| 쌍승 M-R → M-S2 | 0.1230 | [0.1224, 0.1240] | 8/8 |
| 삼쌍승 M-R → M-S2 | 0.1069 | [0.1054, 0.1082] | 8/8 |
| 삼쌍승 M-R → M-S3 | 0.1554 | [0.1534, 0.1573] | 8/8 |
| 삼쌍승 M-S2 → M-S3 | 0.0498 | [0.0486, 0.0509] | 8/8 |

다만 M-R의 모든 가중함수 인수가 단승 공통지지에 들어오는 조합 비율은 쌍승
69.5%, 삼쌍승 38.1%뿐이다. M-S2는 약 98.0%, M-S3은 약 98.9%다. 따라서
M-R의 절대 성능은 꼬리 외삽에 더 민감하며, 강한 결론은 함수형 하나가 아니라
isotonic/Prelec 방향 일치와 power 음성대조를 함께 보고해야 한다.

## 재현 명령

```bash
python -m analysis.behavioral_analysis
python -m analysis.behavioral_figures
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
- `outputs/behavioral_analysis_manifest.json`
- `figures/calibration-rank-probabilities.pdf`
- `figures/model-comparison-behavioral.pdf`
- `figures/model-comparison-support.pdf`

70MB 내외의 `outputs/behavioral_model_metrics_by_race.csv`는 결정론적 재생성
산출물이므로 Git에서 제외한다.
