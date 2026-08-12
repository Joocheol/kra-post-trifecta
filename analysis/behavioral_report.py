#!/usr/bin/env python3
"""Generate manuscript-facing LaTeX tables from frozen behavioral CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MARKET_LABELS = {
    "exacta": "쌍승",
    "trifecta": "삼쌍승",
    "quinella": "복승",
    "trio": "삼복승",
}
MARKET_ORDER = {name: index for index, name in enumerate(MARKET_LABELS)}
MODEL_ORDER = {"M-R": 0, "M-S2": 1, "M-S3": 2}
BENCHMARK_ORDER = {
    "raw_harville": 0,
    "discounted_harville": 1,
    "M-R": 2,
    "M-S2": 3,
    "M-S3": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))
    return parser.parse_args()


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rank_table(rank: pd.DataFrame) -> list[str]:
    required = {
        "probability_model",
        "stage",
        "n_validation_races",
        "mean_log_loss",
        "mean_fold_ece_10",
    }
    if not required.issubset(rank.columns):
        raise ValueError(f"rank table lacks columns: {sorted(required - set(rank.columns))}")
    labels = {"harville": "Harville", "stage_temperature": "단계조정"}
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "단계 & 확률모형 & 검증경주 & 평균 로그손실 & 평균 fold ECE \\\\",
        "\\midrule",
    ]
    for _, row in rank.sort_values(["stage", "probability_model"]).iterrows():
        stage = {1: "1위", 2: "2위$\\mid$1위", 3: "3위$\\mid$1·2위"}[int(row["stage"])]
        lines.append(
            f"{stage} & {labels[row['probability_model']]} & "
            f"{int(row['n_validation_races']):,} & {row['mean_log_loss']:.4f} & "
            f"{100 * row['mean_fold_ece_10']:.2f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return lines


def comparison_table(comparison: pd.DataFrame) -> list[str]:
    data = comparison[
        comparison["probability_model"].eq("stage_temperature")
        & comparison["tail_model"].eq("prelec")
        & comparison["price_model"].isin(MODEL_ORDER)
    ].copy()
    present = set(data["target_market"])
    if present != set(MARKET_LABELS):
        raise ValueError(f"behavioral comparison markets differ: {sorted(present)}")
    data["market_order"] = data["target_market"].map(MARKET_ORDER)
    data["model_order"] = data["price_model"].map(MODEL_ORDER)
    data = data.sort_values(["market_order", "model_order"])
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "시장 & 모형 & 경주수 & 중앙 TV & 중앙 로그 RMSE & 공통지지 \\\\",
        "\\midrule",
    ]
    previous = None
    for _, row in data.iterrows():
        market = row["target_market"]
        if previous is not None and market != previous:
            lines.append("\\addlinespace")
        lines.append(
            f"{MARKET_LABELS[market]} & {row['price_model']} & {int(row['n_races']):,} & "
            f"{row['median_tv']:.4f} & {row['median_log_rmse']:.4f} & "
            f"{100 * row['mean_support_share']:.1f}\\% \\\\"
        )
        previous = market
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return lines


def improvement_table(improvements: pd.DataFrame) -> list[str]:
    data = improvements[
        improvements["probability_model"].eq("stage_temperature")
        & improvements["tail_model"].eq("prelec")
        & improvements["loss"].eq("tv")
    ].copy()
    data["market_order"] = data["target_market"].map(MARKET_ORDER)
    data["baseline_order"] = data["baseline_model"].map(MODEL_ORDER)
    data["challenger_order"] = data["challenger_model"].map(MODEL_ORDER)
    data = data.sort_values(["market_order", "baseline_order", "challenger_order"])
    if len(data) != 8:
        raise ValueError(f"expected eight preferred TV contrasts, found {len(data)}")
    lines = [
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "시장 & 기준 & 비교모형 & TV 개선 중앙값 & 95\\% 구간 & 양의 연도 \\\\",
        "\\midrule",
    ]
    previous = None
    for _, row in data.iterrows():
        market = row["target_market"]
        if previous is not None and market != previous:
            lines.append("\\addlinespace")
        interval = (
            f"[{row['bootstrap_median_ci_lower']:.4f}, "
            f"{row['bootstrap_median_ci_upper']:.4f}]"
        )
        lines.append(
            f"{MARKET_LABELS[market]} & {row['baseline_model']} & "
            f"{row['challenger_model']} & {row['median_loss_reduction']:.4f} & "
            f"{interval} & {int(row['years_with_positive_median'])}/"
            f"{int(row['n_years'])} \\\\"
        )
        previous = market
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return lines


def same_sample_table(comparison: pd.DataFrame) -> list[str]:
    required = {
        "target_market",
        "price_model",
        "n_races",
        "median_tv",
        "median_common_tv",
        "median_trimmed_tv",
        "mean_common_support_share",
    }
    if not required.issubset(comparison.columns):
        raise ValueError(
            f"same-sample table lacks columns: {sorted(required - set(comparison.columns))}"
        )
    labels = {
        "raw_harville": "원시 Harville",
        "discounted_harville": "단계조정 Harville",
        "M-R": "M-R",
        "M-S2": "M-S2",
        "M-S3": "M-S3",
    }
    data = comparison.copy()
    data["market_order"] = data["target_market"].map(MARKET_ORDER)
    data["model_order"] = data["price_model"].map(BENCHMARK_ORDER)
    data = data.sort_values(["market_order", "model_order"])
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "시장 & 모형 & 경주수 & 전체 TV & 공통지지 TV & 꼬리제거 TV & 공통지지 \\\\",
        "\\midrule",
    ]
    previous = None
    for _, row in data.iterrows():
        market = row["target_market"]
        if previous is not None and market != previous:
            lines.append("\\addlinespace")
        lines.append(
            f"{MARKET_LABELS[market]} & {labels[row['price_model']]} & "
            f"{int(row['n_races']):,} & {row['median_tv']:.4f} & "
            f"{row['median_common_tv']:.4f} & {row['median_trimmed_tv']:.4f} & "
            f"{100 * row['mean_common_support_share']:.1f}\\% \\\\"
        )
        previous = market
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return lines


def main() -> None:
    args = parse_args()
    rank = pd.read_csv(args.output_dir / "rank_probability_validation.csv")
    comparison = pd.read_csv(args.output_dir / "behavioral_model_comparison.csv")
    improvements = pd.read_csv(args.output_dir / "behavioral_model_improvements.csv")
    same_sample = pd.read_csv(
        args.output_dir / "behavioral_same_sample_benchmarks.csv"
    )
    time_forward = pd.read_csv(
        args.output_dir / "behavioral_time_forward_improvements.csv"
    )
    _write(args.table_dir / "behavioral_rank_validation.tex", rank_table(rank))
    _write(args.table_dir / "behavioral_model_comparison.tex", comparison_table(comparison))
    _write(args.table_dir / "behavioral_model_improvements.tex", improvement_table(improvements))
    _write(
        args.table_dir / "behavioral_same_sample_benchmarks.tex",
        same_sample_table(same_sample),
    )
    _write(args.table_dir / "behavioral_time_forward.tex", improvement_table(time_forward))


if __name__ == "__main__":
    main()
