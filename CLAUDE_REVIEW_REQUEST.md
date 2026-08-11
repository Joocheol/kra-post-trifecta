# Claude 재검토 요청: PR #6 P3 Paper CI 계약 문서화

## 상태와 범위

주분석은 PR #3에서 병합됐고, PR #5는 P3 공유가격 MILP의 종료상태 처리와
동결표 재현성을 강화했다. 이번 PR #6은 PR #5 병합 뒤 남은 설명상 권고만
정리하는 문서화 PR이다.

실질 변경은 다음 두 파일에 한정된다.

- `.github/workflows/paper-ci.yml`: 기존 P3 검증 로직을 Contract 1--3으로
  구분해 설명하는 주석
- `CLAUDE_REVIEW_REQUEST.md`: 이번 PR의 실제 검토 범위를 지정하는 문서

분석 코드, 허용오차 값, 동결 산출물, 표, 그림, 원고, 표본, 식별전략,
threshold와 주결론은 변경하지 않는다. 이번 검토는 전체 연구를 다시 설계하거나
행동모형을 평가하기 위한 것이 아니라, 새 설명이 이미 존재하는 CI 로직과
동결 산출물에 정확히 대응하는지 확인하기 위한 것이다.

## P3 Paper CI의 세 계약

1. Contract 1은 분석 명령이 새로 만든 CSV와 같은 실행이 저장한 TeX를
   현재 renderer로 직접 대조한다. 이는 live pipeline 연결을 검사한다.
2. Contract 2는 동결된 manuscript-facing CSV와 TeX만 현재 renderer로
   대조한다. MILP를 다시 풀지 않으며 solver time limit의 재현 여부를 주장하지
   않는다.
3. Contract 3은 fresh와 frozen 수치 endpoint를 비교한다. sharp 행은 엄격한
   수치 재현을 요구하고, 한쪽이 non-sharp이면 수학적 인증과 다른 쪽 해와의
   호환성을 검사한다.

동결된 n=16 대표경주는 lower-certified다. 따라서 fresh 실행도 non-sharp이면
양쪽이 non-sharp인 분기가 실제로 실행된다. 그 분기의 `atol=5e-4`는 인증된
비첨예 endpoint를 비교하기 위한 보수적 CI 허용오차다. 반복 solver 실행에서
관측한 변동폭을 추정한 값이 아니며, reporting precision을 뜻하지도 않는다.
이 값은 n=16의 약 0.03 폭 인증구간보다 작지만 표의 소수 넷째 자리에는 영향을
줄 수 있다.

## 이번 재검토의 핵심 질문

1. Contract 1--3의 설명이 실제 인라인 Python 분기와 정확히 일치하는가?
2. n=16 frozen 행의 lower-certified 상태와 양쪽 non-sharp 분기의 실행조건을
   사실대로 설명하는가?
3. `5e-4`를 경험적으로 측정된 solver 안정성이나 보고 정밀도로 과장하지
   않고, 보수적 CI 비교 허용오차로 정확히 한정하는가?
4. Contract 2가 solver를 실행하거나 time limit 재현을 요구한다고 오해할
   여지가 없는가?
5. 이번 diff에 분석 로직, 허용오차 값, 결과, 원고 또는 연구결론의 변경이
   숨어 있지 않은가?
6. workflow 파일의 구문, heredoc 들여쓰기와 파일 말미 개행이 유효한가?

## 검토 태도와 출력

코드의 기존 계약이 타당하다는 사실을 가정하지 말고, 새 주석을 실제 분기,
`outputs/main_order_information_joint.csv`,
`tables/main_order_information_joint.tex`, 관련 시험과 대조한다. 다만 이번
PR이 변경하지 않은 연구설계와 행동모형을 새 쟁점으로 확장하지 않는다.

저장소의 검증된 JSON 형식을 사용한다. 구체적 오류는 findings에 기록하고,
저자 판단이 필요한 사항만 author_questions에 둔다. 파일을 수정하거나
커밋·푸시·라벨·ready 전환·병합하지 않는다.
