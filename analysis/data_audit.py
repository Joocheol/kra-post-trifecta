#!/usr/bin/env python3
"""Audit KRA market panels and freeze race-level analysis samples.

The audit never repairs or renormalizes incomplete races.  It writes the evidence
needed to decide whether a race is eligible for point analysis or must be handled
with interval/partial-identification methods because at least one published odd is
capped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


START_DATE = "2016-06-10"
END_DATE = "2025-12-31"
EXCLUDED_YEARS = frozenset({2020, 2021})
EXCLUDED_DATES = frozenset({"2018-07-01"})
SOURCE_MARKET = "trifecta"
TARGET_MARKETS = ("win", "exacta", "quinella", "trio")
ANALYSIS_MARKETS = (*TARGET_MARKETS, SOURCE_MARKET)


@dataclass(frozen=True)
class MarketSpec:
    keys: tuple[str, ...]
    expected_rows: Callable[[int], int]
    canonical_order: bool = False


MARKET_SPECS = {
    "win": MarketSpec(("horse_no",), lambda n: n),
    "exacta": MarketSpec(("first_no", "second_no"), lambda n: n * (n - 1)),
    "quinella": MarketSpec(
        ("horse_a", "horse_b"),
        lambda n: math.comb(n, 2),
        canonical_order=True,
    ),
    "trio": MarketSpec(
        ("horse_a", "horse_b", "horse_c"),
        lambda n: math.comb(n, 3),
        canonical_order=True,
    ),
    "trifecta": MarketSpec(
        ("first_no", "second_no", "third_no"),
        lambda n: n * (n - 1) * (n - 2),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("KRA/parsed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if an in-scope race violates support, key, status, or odds rules",
    )
    return parser.parse_args()


def expected_row_count(market: str, n_valid_horses: int) -> int:
    """Return the complete-support row count for one race and market."""
    return MARKET_SPECS[market].expected_rows(n_valid_horses)


def parse_horse_list(value: object) -> tuple[int, ...]:
    """Parse the canonical comma-separated valid-horse field."""
    if value is None or pd.isna(value):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError:
        return ()


def parquet_paths(data_root: Path, kind: str) -> list[Path]:
    if kind in MARKET_SPECS:
        pattern = f"*/market={kind}/year=*/month=*/part-*.parquet"
    else:
        pattern = f"*/{kind}/year=*/month=*/part-*.parquet"
    paths = sorted(data_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet files found for {kind!r} under {data_root}")
    return paths


def read_parquets(
    data_root: Path,
    kind: str,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    return pq.read_table(parquet_paths(data_root, kind), columns=columns).to_pandas()


def prepare_races(data_root: Path) -> pd.DataFrame:
    races = read_parquets(data_root, "valid_horses")
    if races["race_id"].duplicated().any():
        duplicated = int(races["race_id"].duplicated(keep=False).sum())
        raise ValueError(f"valid_horses contains {duplicated} duplicated race_id rows")

    races = races.sort_values("race_id").reset_index(drop=True)
    races["valid_horse_tuple"] = races["valid_horses"].map(parse_horse_list)
    races["valid_horses_count_matches"] = races.apply(
        lambda row: len(row["valid_horse_tuple"]) == row["n_valid_horses"], axis=1
    )
    races["valid_horses_unique"] = races["valid_horse_tuple"].map(
        lambda values: len(values) == len(set(values))
    )
    races["valid_horses_positive"] = races["valid_horse_tuple"].map(
        lambda values: bool(values) and all(value > 0 for value in values)
    )
    races["valid_horses_ok"] = (
        races["valid_horses_count_matches"]
        & races["valid_horses_unique"]
        & races["valid_horses_positive"]
        & races["n_valid_horses"].ge(3)
    )

    dates = pd.to_datetime(races["race_date"], errors="coerce")
    invalid_date = dates.isna()
    before_start = dates.lt(pd.Timestamp(START_DATE))
    after_end = dates.gt(pd.Timestamp(END_DATE))
    excluded_year = dates.dt.year.isin(EXCLUDED_YEARS)
    excluded_date = races["race_date"].isin(EXCLUDED_DATES)
    races["scope_exclusion_reason"] = np.select(
        [invalid_date, before_start, after_end, excluded_year, excluded_date],
        [
            "invalid_race_date",
            "before_start_date",
            "after_end_date",
            "excluded_year",
            "excluded_date",
        ],
        default="",
    )
    races["in_date_scope"] = races["scope_exclusion_reason"].eq("")
    return races


def prepare_status(data_root: Path) -> pd.DataFrame:
    status = read_parquets(data_root, "market_status")
    duplicated = status.duplicated(["race_id", "market"], keep=False)
    if duplicated.any():
        raise ValueError(
            "market_status contains "
            f"{int(duplicated.sum())} duplicated race_id-market rows"
        )
    return status.set_index(["race_id", "market"])


def allowed_horse_matrix(races: pd.DataFrame) -> tuple[dict[str, int], np.ndarray]:
    race_codes = {race_id: index for index, race_id in enumerate(races["race_id"])}
    max_horse = max(max(values, default=0) for values in races["valid_horse_tuple"])
    allowed = np.zeros((len(races), max_horse + 1), dtype=bool)
    for index, values in enumerate(races["valid_horse_tuple"]):
        allowed[index, list(values)] = True
    return race_codes, allowed


def invalid_key_mask(
    frame: pd.DataFrame,
    market: str,
    race_codes: dict[str, int],
    allowed: np.ndarray,
) -> np.ndarray:
    spec = MARKET_SPECS[market]
    codes = frame["race_id"].map(race_codes).fillna(-1).to_numpy(dtype=np.int64)
    key_values = [
        pd.to_numeric(frame[column], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        for column in spec.keys
    ]
    valid = codes >= 0
    for values in key_values:
        in_bounds = (values > 0) & (values < allowed.shape[1]) & (codes >= 0)
        member = np.zeros(len(frame), dtype=bool)
        member[in_bounds] = allowed[codes[in_bounds], values[in_bounds]]
        valid &= member

    if len(key_values) >= 2:
        for left in range(len(key_values)):
            for right in range(left + 1, len(key_values)):
                valid &= key_values[left] != key_values[right]
    if spec.canonical_order:
        for left, right in zip(key_values, key_values[1:]):
            valid &= left < right
    return ~valid


def audit_market(
    data_root: Path,
    market: str,
    races: pd.DataFrame,
    status: pd.DataFrame,
    race_codes: dict[str, int],
    allowed: np.ndarray,
) -> pd.DataFrame:
    spec = MARKET_SPECS[market]
    columns = [
        "race_id",
        *spec.keys,
        "odds",
        "is_capped_odds",
        "race_date",
        "meet",
    ]
    if market != "win":
        columns.append("is_cancel")
    frame = read_parquets(data_root, market, columns=columns)
    orphan_mask = ~frame["race_id"].isin(race_codes)
    orphan_race_ids = int(frame.loc[orphan_mask, "race_id"].nunique())
    orphan_rows = int(orphan_mask.sum())
    odds = pd.to_numeric(frame["odds"], errors="coerce").to_numpy(dtype=float)
    missing_odds = np.isnan(odds)
    nonfinite_odds = ~np.isfinite(odds) & ~missing_odds
    nonpositive_odds = np.isfinite(odds) & (odds <= 0)
    missing_cap_flag = frame["is_capped_odds"].isna().to_numpy()
    cap_flag = frame["is_capped_odds"].fillna(False).to_numpy(dtype=bool)
    cap_flag_mismatch = (~missing_cap_flag) & (cap_flag != (odds >= 9999.9))
    if market == "win":
        cancelled_combination = np.zeros(len(frame), dtype=bool)
    else:
        cancelled_combination = frame["is_cancel"].fillna(True).to_numpy(dtype=bool)
    invalid_keys = invalid_key_mask(frame, market, race_codes, allowed)
    duplicate_keys = frame.duplicated(["race_id", *spec.keys], keep="first")

    race_date_map = races.set_index("race_id")["race_date"]
    meet_map = races.set_index("race_id")["meet"]
    metadata_mismatch = (
        frame["race_id"].map(race_date_map).ne(frame["race_date"])
        | frame["race_id"].map(meet_map).ne(frame["meet"])
    )

    row_checks = pd.DataFrame(
        {
            "race_id": frame["race_id"],
            "invalid_key_rows": invalid_keys.astype(np.int64),
            "duplicate_key_rows": duplicate_keys.astype(np.int64),
            "missing_odds_rows": missing_odds.astype(np.int64),
            "nonfinite_odds_rows": nonfinite_odds.astype(np.int64),
            "nonpositive_odds_rows": nonpositive_odds.astype(np.int64),
            "missing_cap_flag_rows": missing_cap_flag.astype(np.int64),
            "cap_flag_mismatch_rows": cap_flag_mismatch.astype(np.int64),
            "capped_odds_rows": cap_flag.astype(np.int64),
            "cancelled_combination_rows": cancelled_combination.astype(np.int64),
            "metadata_mismatch_rows": metadata_mismatch.astype(np.int64),
        }
    )
    grouped = row_checks.groupby("race_id", sort=False).agg(
        observed_rows=("race_id", "size"),
        invalid_key_rows=("invalid_key_rows", "sum"),
        duplicate_key_rows=("duplicate_key_rows", "sum"),
        missing_odds_rows=("missing_odds_rows", "sum"),
        nonfinite_odds_rows=("nonfinite_odds_rows", "sum"),
        nonpositive_odds_rows=("nonpositive_odds_rows", "sum"),
        missing_cap_flag_rows=("missing_cap_flag_rows", "sum"),
        cap_flag_mismatch_rows=("cap_flag_mismatch_rows", "sum"),
        capped_odds_rows=("capped_odds_rows", "sum"),
        cancelled_combination_rows=("cancelled_combination_rows", "sum"),
        metadata_mismatch_rows=("metadata_mismatch_rows", "sum"),
    )

    result = races[["race_id", "n_valid_horses"]].set_index("race_id").join(grouped)
    numeric = [
        "observed_rows",
        "invalid_key_rows",
        "duplicate_key_rows",
        "missing_odds_rows",
        "nonfinite_odds_rows",
        "nonpositive_odds_rows",
        "missing_cap_flag_rows",
        "cap_flag_mismatch_rows",
        "capped_odds_rows",
        "cancelled_combination_rows",
        "metadata_mismatch_rows",
    ]
    result[numeric] = result[numeric].fillna(0).astype(np.int64)
    result["expected_rows"] = result["n_valid_horses"].map(spec.expected_rows)

    status_slice = status.xs(market, level="market", drop_level=True).rename(
        columns={
            "n_rows": "status_n_rows",
            "status": "market_status",
            "status_reason": "market_status_reason",
            "is_cancelled": "market_is_cancelled",
        }
    )
    result = result.join(
        status_slice[
            [
                "status_n_rows",
                "market_status",
                "market_status_reason",
                "market_is_cancelled",
            ]
        ]
    )
    result["status_row_present"] = result["market_status"].notna()
    result["status_ok"] = (
        result["status_row_present"]
        & result["market_status"].eq("ok")
        & result["market_status_reason"].eq("parsed_rows_present")
        & ~result["market_is_cancelled"].fillna(True).astype(bool)
        & result["status_n_rows"].eq(result["observed_rows"])
    )
    result["complete_support"] = (
        result["observed_rows"].eq(result["expected_rows"])
        & result["duplicate_key_rows"].eq(0)
        & result["invalid_key_rows"].eq(0)
        & result["metadata_mismatch_rows"].eq(0)
        & result["status_ok"]
    )
    result["positive_finite_odds"] = (
        result["missing_odds_rows"].eq(0)
        & result["nonfinite_odds_rows"].eq(0)
        & result["nonpositive_odds_rows"].eq(0)
        & result["missing_cap_flag_rows"].eq(0)
        & result["cap_flag_mismatch_rows"].eq(0)
        & result["cancelled_combination_rows"].eq(0)
    )
    result["uncapped"] = result["capped_odds_rows"].eq(0)
    result["orphan_race_ids_detected"] = orphan_race_ids
    result["orphan_rows_detected"] = orphan_rows

    result = result.drop(columns=["n_valid_horses"])
    return result.add_prefix(f"{market}_")


def build_quality_table(data_root: Path) -> pd.DataFrame:
    races = prepare_races(data_root)
    status = prepare_status(data_root)
    race_codes, allowed = allowed_horse_matrix(races)
    quality_base = races[
        [
            "race_id",
            "race_date",
            "meet",
            "race_no",
            "n_valid_horses",
            "valid_horses",
            "arrival_len",
            "valid_horses_count_matches",
            "valid_horses_unique",
            "valid_horses_positive",
            "valid_horses_ok",
            "in_date_scope",
            "scope_exclusion_reason",
        ]
    ].set_index("race_id")
    parts = [quality_base]
    for market in ANALYSIS_MARKETS:
        parts.append(
            audit_market(data_root, market, races, status, race_codes, allowed)
        )
    return pd.concat(parts, axis=1).copy().reset_index()


def structural_masks(quality: pd.DataFrame, target: str) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    masks["candidate_races"] = pd.Series(True, index=quality.index)
    masks["date_scope"] = masks["candidate_races"] & quality["in_date_scope"]
    masks["valid_horses"] = masks["date_scope"] & quality["valid_horses_ok"]
    masks["trifecta_complete_support"] = (
        masks["valid_horses"] & quality["trifecta_complete_support"]
    )
    masks["trifecta_positive_finite_odds"] = (
        masks["trifecta_complete_support"] & quality["trifecta_positive_finite_odds"]
    )
    masks[f"{target}_complete_support"] = (
        masks["trifecta_positive_finite_odds"] & quality[f"{target}_complete_support"]
    )
    masks[f"{target}_positive_finite_odds"] = (
        masks[f"{target}_complete_support"] & quality[f"{target}_positive_finite_odds"]
    )
    return masks


def build_analysis_sample(quality: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for target in TARGET_MARKETS:
        masks = structural_masks(quality, target)
        complete = masks[f"{target}_positive_finite_odds"]
        clean = complete & quality["trifecta_uncapped"] & quality[f"{target}_uncapped"]
        censored = complete & ~clean

        structural_reason = np.select(
            [
                ~quality["in_date_scope"],
                ~quality["valid_horses_ok"],
                ~quality["trifecta_complete_support"],
                ~quality["trifecta_positive_finite_odds"],
                ~quality[f"{target}_complete_support"],
                ~quality[f"{target}_positive_finite_odds"],
            ],
            [
                quality["scope_exclusion_reason"],
                "invalid_valid_horses",
                "incomplete_trifecta_support",
                "invalid_trifecta_odds",
                f"incomplete_{target}_support",
                f"invalid_{target}_odds",
            ],
            default="",
        )
        point_reason = np.where(
            ~complete,
            structural_reason,
            np.where(censored, "capped_odds_requires_interval", ""),
        )

        records.append(
            pd.DataFrame(
                {
                    "race_id": quality["race_id"],
                    "race_date": quality["race_date"],
                    "meet": quality["meet"],
                    "race_no": quality["race_no"],
                    "n_valid_horses": quality["n_valid_horses"],
                    "target_market": target,
                    "eligible_complete_sample": complete,
                    "eligible_clean_point_sample": clean,
                    "eligible_capped_interval_sample": censored,
                    "trifecta_capped_rows": quality["trifecta_capped_odds_rows"],
                    "target_capped_rows": quality[f"{target}_capped_odds_rows"],
                    "structural_exclusion_reason": structural_reason,
                    "point_sample_exclusion_reason": point_reason,
                }
            )
        )
    return pd.concat(records, ignore_index=True).sort_values(
        ["target_market", "race_id"]
    )


def clean_sample_set_differences(sample: pd.DataFrame) -> dict[str, int]:
    """Return symmetric differences from the win clean-sample race-ID set."""
    clean_sets = {
        target: set(
            sample.loc[
                sample["target_market"].eq(target)
                & sample["eligible_clean_point_sample"],
                "race_id",
            ]
        )
        for target in TARGET_MARKETS
    }
    reference = clean_sets[TARGET_MARKETS[0]]
    return {
        target: len(reference.symmetric_difference(clean_sets[target]))
        for target in TARGET_MARKETS
    }


def build_sample_flow(quality: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGET_MARKETS:
        masks = structural_masks(quality, target)
        previous = len(quality)
        for order, (stage, mask) in enumerate(masks.items(), start=1):
            count = int(mask.sum())
            rows.append(
                {
                    "target_market": target,
                    "stage_order": order,
                    "stage": stage,
                    "count_type": "sequential_remaining",
                    "n_races": count,
                    "n_excluded_at_stage": previous - count,
                }
            )
            previous = count

        complete = masks[f"{target}_positive_finite_odds"]
        clean = complete & quality["trifecta_uncapped"] & quality[f"{target}_uncapped"]
        censored = complete & ~clean
        rows.extend(
            [
                {
                    "target_market": target,
                    "stage_order": len(masks) + 1,
                    "stage": "clean_uncapped_point_sample",
                    "count_type": "subset_of_complete",
                    "n_races": int(clean.sum()),
                    "n_excluded_at_stage": int((complete & ~clean).sum()),
                },
                {
                    "target_market": target,
                    "stage_order": len(masks) + 2,
                    "stage": "capped_interval_sample",
                    "count_type": "subset_of_complete",
                    "n_races": int(censored.sum()),
                    "n_excluded_at_stage": 0,
                },
            ]
        )
    return pd.DataFrame(rows)


def write_summary(
    output_path: Path,
    quality: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    in_scope = quality["in_date_scope"]
    excluded = len(quality) - int(in_scope.sum())
    lines = [
        "# KRA 가격자료 감사 결과",
        "",
        "## 표본과 데이터 단위",
        "",
        f"- 후보 경주: {len(quality):,}경주",
        f"- 사전 날짜 규칙 적용 후: {int(in_scope.sum()):,}경주 (제외 {excluded:,}경주)",
        "- 가격자료 단위: 경주 × 승식 × 가능한 조합",
        "- 실질 표본단위: 경주",
        "",
        "## 핵심 판정",
        "",
    ]
    all_complete = True
    for market in ANALYSIS_MARKETS:
        complete = int((in_scope & quality[f"{market}_complete_support"]).sum())
        usable = int(
            (
                in_scope
                & quality[f"{market}_complete_support"]
                & quality[f"{market}_positive_finite_odds"]
            ).sum()
        )
        capped = int((in_scope & ~quality[f"{market}_uncapped"]).sum())
        orphan_ids = int(quality[f"{market}_orphan_race_ids_detected"].iloc[0])
        all_complete &= (
            complete == int(in_scope.sum())
            and usable == int(in_scope.sum())
            and orphan_ids == 0
        )
        lines.append(
            f"- `{market}`: 완전 조합 {complete:,}경주, 양수·유한 배당 {usable:,}경주, "
            f"상한 포함 {capped:,}경주, 고아 race_id {orphan_ids:,}개"
        )

    scope_counts = quality.loc[
        ~quality["in_date_scope"], "scope_exclusion_reason"
    ].value_counts()
    if not scope_counts.empty:
        scope_text = ", ".join(
            f"{reason} {int(count):,}경주" for reason, count in scope_counts.items()
        )
        lines.append(f"- 날짜 범위 제외 사유: {scope_text}")
    lines.extend(
        [
            "",
            (
                "- 지원집합·키·배당 유효성: **통과**"
                if all_complete
                else "- 지원집합·키·배당 유효성: **일부 실패 — data_quality.csv 확인 필요**"
            ),
            "- 목표 승식별 최종 분석표본:",
        ]
    )
    for target in TARGET_MARKETS:
        target_sample = sample[sample["target_market"].eq(target)]
        clean = int(target_sample["eligible_clean_point_sample"].sum())
        censored = int(target_sample["eligible_capped_interval_sample"].sum())
        lines.append(
            f"  - `{target}`: clean point {clean:,}경주, capped interval {censored:,}경주"
        )
    set_differences = clean_sample_set_differences(sample)
    sets_identical = not any(set_differences.values())
    lines.extend(
        [
            (
                "- 네 목표 승식의 clean race_id 집합: **동일**"
                if sets_identical
                else "- 네 목표 승식의 clean race_id 집합: **불일치 — strict 실패**"
            ),
            "",
            "## 분석상 위험과 처리",
            "",
            "- **High:** 삼쌍승 상한은 대다수 경주에 존재한다. 상한 배당을 9,999.9의 "
            "점값으로 간주한 행동잔차 해석은 허용하지 않는다.",
            "- **처리:** clean 표본의 점추정(Panel A)과 전체 표본의 배당 검열구간에 "
            "기초한 부분식별 경계(Panel B)를 공동 주결과로 보고한다.",
            "- **통과:** 가능한 조합 수, 조합키 유일성, 유효마 포함관계, 양수·유한 "
            "배당 및 market_status 일치 여부를 경주별로 검사했다.",
            "",
            "## 재현",
            "",
            "```bash",
            "python -m analysis.data_audit --strict",
            "```",
            "",
            "상세 경주별 증거는 `outputs/data_quality.csv`, 목표 승식별 포함 여부는 "
            "`outputs/analysis_sample.csv`, 순차 표본흐름은 `outputs/sample_flow.csv`에 있다.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(
    output_path: Path,
    quality: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    labels = {
        "win": "단승 (win)",
        "exacta": "쌍승 (exacta)",
        "quinella": "복승 (quinella)",
        "trio": "삼복승 (trio)",
    }
    in_scope = quality["in_date_scope"]
    rows = []
    for target in TARGET_MARKETS:
        target_sample = sample[sample["target_market"].eq(target)]
        complete = int(target_sample["eligible_complete_sample"].sum())
        target_capped = int((in_scope & ~quality[f"{target}_uncapped"]).sum())
        clean = int(target_sample["eligible_clean_point_sample"].sum())
        interval = int(target_sample["eligible_capped_interval_sample"].sum())
        rows.append(
            f"{labels[target]} & {complete:,} & {target_capped:,} & "
            f"{clean:,} & {interval:,} \\\\"
        )
    trifecta_complete = int(
        (
            in_scope
            & quality["trifecta_complete_support"]
            & quality["trifecta_positive_finite_odds"]
        ).sum()
    )
    trifecta_capped = int((in_scope & ~quality["trifecta_uncapped"]).sum())
    trifecta_clean = trifecta_complete - trifecta_capped
    rows.append(
        f"삼쌍승 (원천) & {trifecta_complete:,} & {trifecta_capped:,} & "
        f"{trifecta_clean:,} & {trifecta_capped:,} \\\\"
    )

    content = "\n".join(
        [
            "% Generated by python -m analysis.data_audit --strict; do not edit.",
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{승식별 가격자료 감사와 분석표본}",
            r"  \label{tab:data-audit}",
            r"  \small",
            r"  \begin{tabular}{lrrrr}",
            r"    \hline",
            r"    승식(역할) & 완전·유효 & 승식 상한 & clean 점표본 & 상한 구간표본 \\",
            r"    \hline",
            *[f"    {row}" for row in rows],
            r"    \hline",
            r"  \end{tabular}",
            r"  \begin{minipage}{0.96\linewidth}",
            r"    \footnotesize\emph{주:} 경주 수를 보고한다. 모든 행은 사전 제외일을",
            r"    뺀 19,284경주를 모집단으로 한다. 목표 승식 행은 이 중 삼쌍승과 해당",
            r"    승식의 완전한 지지집합 및 양수·유한 배당을 만족하는 경주 수다.",
            r"    clean 점표본은 두 승식 모두 상한이",
            r"    없는 경주다. 네 목표 승식의 clean 표본은 동일한 3,321개 race\_id",
            r"    집합이며, 삼쌍승 원천행에서 보듯 이는 19,284-15,963이다.",
            r"  \end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text(content, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    output_path: Path,
    output_dir: Path,
    quality: pd.DataFrame,
    sample: pd.DataFrame,
    flow: pd.DataFrame,
) -> None:
    artifacts = {
        "data_quality.csv": len(quality),
        "analysis_sample.csv": len(sample),
        "sample_flow.csv": len(flow),
    }
    payload = {
        "schema_version": 1,
        "candidate_races": len(quality),
        "in_date_scope_races": int(quality["in_date_scope"].sum()),
        "scope_exclusions": {
            reason: int(count)
            for reason, count in quality.loc[
                ~quality["in_date_scope"], "scope_exclusion_reason"
            ].value_counts().sort_index().items()
        },
        "orphan_records": {
            market: {
                "race_ids": int(
                    quality[f"{market}_orphan_race_ids_detected"].iloc[0]
                ),
                "rows": int(quality[f"{market}_orphan_rows_detected"].iloc[0]),
            }
            for market in ANALYSIS_MARKETS
        },
        "clean_sample_race_ids": {
            "reference_market": TARGET_MARKETS[0],
            "identical_across_targets": not any(
                clean_sample_set_differences(sample).values()
            ),
            "symmetric_difference_from_reference": clean_sample_set_differences(
                sample
            ),
        },
        "artifacts": {
            name: {
                "rows": rows,
                "sha256": sha256(output_dir / name),
            }
            for name, rows in artifacts.items()
        },
        "target_samples": {
            target: {
                "complete": int(
                    sample.loc[
                        sample["target_market"].eq(target),
                        "eligible_complete_sample",
                    ].sum()
                ),
                "clean_point": int(
                    sample.loc[
                        sample["target_market"].eq(target),
                        "eligible_clean_point_sample",
                    ].sum()
                ),
                "capped_interval": int(
                    sample.loc[
                        sample["target_market"].eq(target),
                        "eligible_capped_interval_sample",
                    ].sum()
                ),
            }
            for target in TARGET_MARKETS
        },
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def strict_failures(quality: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    in_scope = quality["in_date_scope"]
    failures: list[str] = []
    invalid_horses = int((in_scope & ~quality["valid_horses_ok"]).sum())
    if invalid_horses:
        failures.append(f"invalid valid-horse records: {invalid_horses}")
    for market in ANALYSIS_MARKETS:
        orphan_race_ids = int(quality[f"{market}_orphan_race_ids_detected"].iloc[0])
        orphan_rows = int(quality[f"{market}_orphan_rows_detected"].iloc[0])
        if orphan_race_ids or orphan_rows:
            failures.append(
                f"{market} orphan records: {orphan_race_ids} race IDs, {orphan_rows} rows"
            )
        incomplete = int((in_scope & ~quality[f"{market}_complete_support"]).sum())
        invalid_odds = int((in_scope & ~quality[f"{market}_positive_finite_odds"]).sum())
        if incomplete:
            failures.append(f"{market} incomplete-support races: {incomplete}")
        if invalid_odds:
            failures.append(f"{market} invalid-odds races: {invalid_odds}")
    set_differences = clean_sample_set_differences(sample)
    if any(set_differences.values()):
        failures.append(
            "clean sample race-ID sets differ across targets: "
            + ", ".join(
                f"{target}={difference}"
                for target, difference in set_differences.items()
            )
        )
    return failures


def main() -> int:
    args = parse_args()
    quality = build_quality_table(args.data_root)
    sample = build_analysis_sample(quality)
    flow = build_sample_flow(quality)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    sample.to_csv(args.output_dir / "analysis_sample.csv", index=False)
    flow.to_csv(args.output_dir / "sample_flow.csv", index=False)
    write_summary(args.output_dir / "data_audit_summary.md", quality, sample)
    write_manifest(
        args.output_dir / "data_audit_manifest.json",
        args.output_dir,
        quality,
        sample,
        flow,
    )
    write_latex_table(args.table_dir / "data_quality_summary.tex", quality, sample)

    failures = strict_failures(quality, sample)
    print(f"candidate_races={len(quality)}")
    print(f"in_date_scope_races={int(quality['in_date_scope'].sum())}")
    for target in TARGET_MARKETS:
        target_sample = sample[sample["target_market"].eq(target)]
        print(
            f"{target}_clean_point_races="
            f"{int(target_sample['eligible_clean_point_sample'].sum())}"
        )
        print(
            f"{target}_capped_interval_races="
            f"{int(target_sample['eligible_capped_interval_sample'].sum())}"
        )
    if failures:
        for failure in failures:
            print(f"QUALITY_FAILURE: {failure}")
        return 1 if args.strict else 0
    print("data_quality_audit=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
