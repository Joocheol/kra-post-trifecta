#!/usr/bin/env python3
"""Render the frozen behavioral-analysis figures from compact CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INK = "#25313c"
GRID = "#d9dee3"
BLUE = "#2f6f9f"
ORANGE = "#d17a22"
OLIVE = "#7b8b3a"
GREY = "#8a9299"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "Joocheol/kra-post-trifecta behavioral_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rank_probability_figure(rank: pd.DataFrame, path: Path) -> None:
    labels = {"harville": "Harville", "stage_temperature": "Stage-adjusted"}
    colors = {"harville": GREY, "stage_temperature": BLUE}
    hatches = {"harville": "///", "stage_temperature": ""}
    stages = [1, 2, 3]
    x = np.arange(len(stages), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    for model_index, model in enumerate(("harville", "stage_temperature")):
        values = rank[rank["probability_model"].eq(model)].set_index("stage")
        offset = (model_index - 0.5) * width
        axes[0].bar(
            x + offset,
            values.loc[stages, "mean_log_loss"],
            width,
            label=labels[model],
            color=colors[model],
            edgecolor=INK,
            linewidth=0.6,
            hatch=hatches[model],
        )
        axes[1].bar(
            x + offset,
            100 * values.loc[stages, "mean_fold_ece_10"],
            width,
            color=colors[model],
            edgecolor=INK,
            linewidth=0.6,
            hatch=hatches[model],
        )
    axes[0].set_title("Out-of-year rank probability: log loss", loc="left")
    axes[0].set_ylabel("Mean negative log score")
    axes[1].set_title("Out-of-year rank probability: calibration", loc="left")
    axes[1].set_ylabel("Mean fold ECE (percentage points)")
    for axis in axes:
        axis.set_xticks(x, ["1st", "2nd | 1st", "3rd | 1st, 2nd"])
        axis.set_xlabel("Rank stage")
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[0].legend().remove()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
    )
    fig.suptitle(
        "Rank-probability validation (19,284 races; eight held-out years)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    save(fig, path)


def behavioral_model_figure(comparison: pd.DataFrame, path: Path) -> None:
    data = comparison[
        comparison["probability_model"].eq("stage_temperature")
        & comparison["tail_model"].eq("prelec")
        & comparison["price_model"].ne("M-U")
    ].copy()
    order = {"M-R": 0, "M-S2": 1, "M-S3": 2}
    colors = {"M-R": GREY, "M-S2": ORANGE, "M-S3": BLUE}
    hatches = {"M-R": "///", "M-S2": "..", "M-S3": ""}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for axis, market, title in zip(
        axes,
        ("exacta", "trifecta"),
        ("Exacta (18,703 races)", "Trifecta (3,321 races)"),
    ):
        panel = data[data["target_market"].eq(market)].copy()
        panel["order"] = panel["price_model"].map(order)
        panel = panel.sort_values("order")
        positions = np.arange(len(panel))
        bars = axis.barh(
            positions,
            panel["median_tv"],
            color=[colors[value] for value in panel["price_model"]],
            edgecolor=INK,
            linewidth=0.6,
        )
        for bar, hatch, value in zip(
            bars,
            [hatches[value] for value in panel["price_model"]],
            panel["median_tv"],
        ):
            bar.set_hatch(hatch)
            axis.text(
                value + 0.007,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=8.5,
            )
        axis.set_yticks(positions, panel["price_model"])
        axis.invert_yaxis()
        axis.set_xlim(0, 0.36)
        axis.set_xlabel("Median race-level total variation distance")
        axis.set_title(title, loc="left")
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Behavioral price-model comparison: Prelec tail specification",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        "M-U is omitted because it is observationally equivalent to reduced M-R; lower is better.",
        fontsize=8,
        color=INK,
    )
    save(fig, path)


def support_figure(comparison: pd.DataFrame, path: Path) -> None:
    data = comparison[
        comparison["probability_model"].eq("stage_temperature")
        & comparison["tail_model"].eq("prelec")
        & comparison["price_model"].isin(["M-R", "M-S2", "M-S3"])
    ].copy()
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    hatches: list[str] = []
    palette = {"M-R": GREY, "M-S2": ORANGE, "M-S3": BLUE}
    hatch_map = {"M-R": "///", "M-S2": "..", "M-S3": ""}
    for market in ("exacta", "trifecta"):
        panel = data[data["target_market"].eq(market)]
        for model in ("M-R", "M-S2", "M-S3"):
            row = panel[panel["price_model"].eq(model)]
            if row.empty:
                continue
            labels.append(f"{market.capitalize()}  {model}")
            values.append(100 * float(row["mean_support_share"].iloc[0]))
            colors.append(palette[model])
            hatches.append(hatch_map[model])
    fig, axis = plt.subplots(figsize=(6.3, 3.0), constrained_layout=True)
    positions = np.arange(len(values))
    bars = axis.barh(
        positions,
        values,
        color=colors,
        edgecolor=INK,
        linewidth=0.6,
    )
    for bar, hatch, value in zip(bars, hatches, values):
        bar.set_hatch(hatch)
        axis.text(
            min(value + 1.0, 101.0),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=8.5,
        )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xlabel("Combination-level share with every weighting argument in win support")
    axis.set_title(
        "Common-support coverage (training-fold 1st-99th percentiles)", loc="left"
    )
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    save(fig, path)


def main() -> None:
    args = parse_args()
    style()
    rank = pd.read_csv(args.output_dir / "rank_probability_validation.csv")
    comparison = pd.read_csv(args.output_dir / "behavioral_model_comparison.csv")
    rank_probability_figure(rank, args.figure_dir / "calibration-rank-probabilities.pdf")
    behavioral_model_figure(
        comparison, args.figure_dir / "model-comparison-behavioral.pdf"
    )
    support_figure(comparison, args.figure_dir / "model-comparison-support.pdf")


if __name__ == "__main__":
    main()
