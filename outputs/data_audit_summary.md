# KRA 가격자료 감사 결과

## 표본과 데이터 단위

- 후보 경주: 19,301경주
- 사전 날짜 규칙 적용 후: 19,284경주 (제외 17경주)
- 가격자료 단위: 경주 × 승식 × 가능한 조합
- 실질 표본단위: 경주

## 핵심 판정

- `win`: 완전 조합 19,284경주, 양수·유한 배당 19,284경주, 상한 포함 0경주, 고아 race_id 0개
- `exacta`: 완전 조합 19,284경주, 양수·유한 배당 19,284경주, 상한 포함 581경주, 고아 race_id 0개
- `quinella`: 완전 조합 19,284경주, 양수·유한 배당 19,284경주, 상한 포함 15경주, 고아 race_id 0개
- `trio`: 완전 조합 19,284경주, 양수·유한 배당 19,284경주, 상한 포함 3,434경주, 고아 race_id 0개
- `trifecta`: 완전 조합 19,284경주, 양수·유한 배당 19,284경주, 상한 포함 15,963경주, 고아 race_id 0개
- 날짜 범위 제외 사유: excluded_date 17경주

- 지원집합·키·배당 유효성: **통과**
- 목표 승식별 최종 분석표본:
  - `win`: clean point 3,321경주, capped interval 15,963경주
  - `exacta`: clean point 3,321경주, capped interval 15,963경주
  - `quinella`: clean point 3,321경주, capped interval 15,963경주
  - `trio`: clean point 3,321경주, capped interval 15,963경주

## 분석상 위험과 처리

- **High:** 삼쌍승 상한은 대다수 경주에 존재한다. 상한 배당을 9,999.9의 점값으로 간주한 행동잔차 해석은 허용하지 않는다.
- **처리:** clean 표본의 점추정(Panel A)과 전체 표본의 배당 검열구간에 기초한 부분식별 경계(Panel B)를 공동 주결과로 보고한다.
- **통과:** 가능한 조합 수, 조합키 유일성, 유효마 포함관계, 양수·유한 배당 및 market_status 일치 여부를 경주별로 검사했다.

## 재현

```bash
python -m analysis.data_audit --strict
```

상세 경주별 증거는 `outputs/data_quality.csv`, 목표 승식별 포함 여부는 `outputs/analysis_sample.csv`, 순차 표본흐름은 `outputs/sample_flow.csv`에 있다.
