# 주분석 구현 계약

이 파일은 `RESEARCH_PLAN.md`의 주분석을 코드로 옮길 때 고정한 구현 세부를 기록한다.
결과를 보기 전에 다음을 고정한다.

- 실행 모듈: `python -m analysis.main_analysis`
- 난수 시드: `20260809`
- 경주 부트스트랩: 999회, 95% percentile CI
- 비상한 1자리 배당의 반올림 envelope: 게시값 ±0.05
- 상한 9,999.9: 실제 총지급배율의 하한으로만 사용하고 상한은 무한대
- 동일 경주 순열 기준: 경주×목표승식×패널마다 시드에서 결정되는 순열 1개
- 타경주 기준: 같은 출전두수에서 자기 경주를 제외한 donor 1개를 시드에서 결정
  - Panel A donor는 clean 표본에서만 선택
  - Panel B donor는 전체 분석표본에서 선택하며 상한 donor이면 가격집합 전체를 전파
  - 같은 출전두수의 다른 경주가 하나도 없으면 그 경주는 타경주 기준의 짝비교에서만 제외하고,
    주모형·Harville·순열·균등 비교와 주표본에서는 유지한다.
- Harville 기준: 단승 정규화 역배당률을 순차적으로 재조정한 1→2→3위 분포
  - 동결 자료에는 단승 상한 경주가 0건이다.
  - 코드가 이를 명시적으로 검사하며, 향후 단승 상한이 생기면 조용히 점값으로 처리하지 않고 실패시킨다.
    그 경우 Harville의 구간화를 별도 방법 변경으로 심의한다.
- Panel A 주집계: 경주균등가중 중앙값
- Panel B: TV exact lower bound와 componentwise extrema를 합한 certified outer upper bound
- 기준모형 우위: Panel A `TV_benchmark-TV_main`, Panel B `TV_lower_benchmark-TV_upper_main`의
  중앙값 bootstrap CI 하한이 0보다 큰 경우에만 인정
- P3 순서정보는 두 패널에 모두 적용한다.
  - Panel A: `(TV_Harville-TV_main)_exacta - (TV_Harville-TV_main)_quinella`
  - Panel B 보수적 하한:
    `L_H,E-U_M,E-U_H,Q+L_M,Q`; 이 경주별 하한의 중앙값 bootstrap CI 하한이 0보다
    큰 경우에만 전체표본에서 양(+)의 순서정보 차이를 지지한다고 판정한다.
- 절대 TV 기준: 0.05, 민감도 0.025와 0.10

Panel B에서 삼쌍승 상태들은 각 목표 승식 결과를 정확히 하나씩 지시하므로 주변화
행렬의 행들은 상태공간의 partition을 이룬다. 따라서 원시 삼쌍승 역배당률 구간을
목표 결과별로 합산한 구간은 `A q`의 허용 가격집합을 정확히 표현한다. 이 축약 뒤
TV 하한 LP는 목표 승식 차원에서만 풀며, 삼쌍승 전체 상태변수를 LP에 직접 넣지 않는다.

`--max-races`는 개발용으로만 사용한다. donor 후보군은 항상 동결된 전체 표본에서 정하므로
개발 부분표본을 바꾸어도 donor 배정 규칙은 달라지지 않는다.
`--skip-bounds`는 개발용 Panel A 점검 옵션이며 최종 산출물에는 사용하지 않는다.

컴팩트 요약 CSV, manifest와 LaTeX 표는 이 PR에서 전체 분석을 실행한 뒤 커밋하여 동결한다.
대용량 경주별 CSV는 재생성 가능 산출물로 유지하고, manifest에 포함된 SHA-256으로 신선도를
검증한다. CI는 동결된 컴팩트 산출물을 재생성 결과와 비교한다.
