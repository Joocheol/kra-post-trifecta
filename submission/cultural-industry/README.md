# 『문화산업연구』 제출 패키지

## 제출 파일

- `kra-cultural-industry-submission-author.hwp`: 저자·소속·이메일 포함 인적사항 원고
- `kra-cultural-industry-submission-blind.hwp`: 저자정보를 제거한 심사용 원고
- `kra-cultural-industry-submission-*-review.pdf`: 같은 본문의 시각 검토용 PDF
- `COVER_LETTER.md`: 편집위원회 제출용 커버레터
- `manuscript.json`: 두 원고의 단일 본문 원천

편집규정에 따라 레터사이즈, 첫 면 단단, 본문 2단, 본문 9pt·줄간격 160을 기준으로 만들었다. HWP는 제공된 학회 샘플의 용지·단 설정과 서체 체계를 사용한다.

## 재생성

```bash
cd submission/cultural-industry
npm install
CULTURAL_INDUSTRY_TEMPLATE=/absolute/path/to/journal-sample.hwp npm run build:hwp
python build-review-pdf.py
```

검토용 PDF는 HWP 변환본이 아니라 동일한 `manuscript.json`에서 생성한 독립 조판본이다. 최종 제출 전 한컴오피스에서 두 HWP를 열어 설치된 HY/휴먼 계열 글꼴과 페이지 번호를 한 번 확인한다.

## 정책 결정

- 매출총액은 관측 변수로 바로잡고 파서의 `turnover_won` 보존을 명시했다.
- 조합별 베팅액과 독립 유효표본크기는 관측되지 않는다는 한계만 유지했다.
- 코드·재현자료는 공개 저장소 약속이 아니라 “합리적 요청 시 제공”으로 통일했다.
- 원시 JSON은 저작권·재배포 범위를 고려해 제공 대상에서 제외했다.
- 전체 국문판은 미게재 기술보고서로 두고 본 원고를 유일한 국문 학술지 게재본으로 정했다.
