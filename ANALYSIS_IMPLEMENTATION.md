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
- Harville 기준: 단승 정규화 역배당률을 순차적으로 재조정한 1→2→3위 분포
- Panel A 주집계: 경주균등가중 중앙값
- Panel B: TV exact lower bound와 componentwise extrema를 합한 certified outer upper bound
- 기준모형 우위: Panel A `TV_benchmark-TV_main`, Panel B `TV_lower_benchmark-TV_upper_main`의
  중앙값 bootstrap CI 하한이 0보다 큰 경우에만 인정
- 절대 TV 기준: 0.05, 민감도 0.025와 0.10

Panel B에서 삼쌍승 상태들은 각 목표 승식 결과를 정확히 하나씩 지시하므로 주변화
행렬의 행들은 상태공간의 partition을 이룬다. 따라서 원시 삼쌍승 역배당률 구간을
목표 결과별로 합산한 구간은 `A q`의 허용 가격집합을 정확히 표현한다. 이 축약 뒤
TV 하한 LP는 목표 승식 차원에서만 풀며, 삼쌍승 전체 상태변수를 LP에 직접 넣지 않는다.

`--max-races`는 CI smoke test 전용이다. 최종 산출물에는 사용하지 않는다.
`--skip-bounds`는 개발용 Panel A 점검 옵션이며 최종 산출물에는 사용하지 않는다.
