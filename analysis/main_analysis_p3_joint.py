"""Sharpness diagnostic for Panel B P3 using a shared-source joint MILP.

The co-primary Panel B decision keeps the pre-registered conservative endpoint
combination. This module does not replace that decision rule. Instead it solves
an exact mixed-integer linear program on one deterministic race per field-size
stratum to quantify how much width is introduced when four TV extrema are
combined separately.

For P3, quinella is a deterministic coarsening of exacta: each unordered first-
two pair is the sum of its two ordered exacta outcomes. Because exacta cells
partition the trifecta state space, the aggregated exacta PriceSet is an exact
representation of all feasible exacta marginals of the source interval box.
Optimizing on that marginal, then coarsening it to quinella, preserves exactly
the shared normalized trifecta price uncertainty relevant for P3 while avoiding
thousands of unnecessary trifecta-state variables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from analysis.main_analysis_core import (
    LP_TOL,
    PriceSet,
    aggregate_point,
    aggregate_price_set,
    harville_trifecta,
    source_group_index,
    stable_uint,
)
from analysis.main_analysis_panels import interval_for_frame, load_market
from analysis.main_analysis_runner import common_race_ids, race_metadata, read_frozen_sample

BIG_M = 2.0


def _scale_bounds(price_set: PriceSet) -> tuple[float, float]:
    lower_sum = float(price_set.lower.sum())
    upper_sum = float(price_set.upper.sum())
    return 1.0 / upper_sum, np.inf if lower_sum <= 0 else 1.0 / lower_sum


def _exacta_to_quinella_map(
    exacta_groups: np.ndarray,
    quinella_groups: np.ndarray,
    edim: int,
    qdim: int,
) -> np.ndarray:
    """Map each exacta outcome to its unique quinella coarsening cell."""
    exacta_groups = np.asarray(exacta_groups, dtype=np.int64)
    quinella_groups = np.asarray(quinella_groups, dtype=np.int64)
    if exacta_groups.shape != quinella_groups.shape:
        raise ValueError("exacta and quinella group maps must have equal length")
    mapping = np.empty(edim, dtype=np.int64)
    for eidx in range(edim):
        qcells = np.unique(quinella_groups[exacta_groups == eidx])
        if len(qcells) != 1:
            raise ValueError("each exacta outcome must map to exactly one quinella outcome")
        mapping[eidx] = int(qcells[0])
    if np.any(mapping < 0) or np.any(mapping >= qdim):
        raise ValueError("invalid exacta-to-quinella coarsening map")
    if len(np.unique(mapping)) != qdim:
        raise ValueError("exacta coarsening does not cover every quinella outcome")
    return mapping


def joint_p3_extrema(
    source_set: PriceSet,
    exacta_groups: np.ndarray,
    exacta_actual: PriceSet,
    exacta_harville: np.ndarray,
    quinella_groups: np.ndarray,
    quinella_actual: PriceSet,
    quinella_harville: np.ndarray,
    *,
    time_limit: float = 120.0,
) -> tuple[float, float]:
    """Return sharp min/max P3 difference under shared source-price uncertainty.

    The optimized quantity is

      TV(A_E,H_E) - TV(A_E,M_E) - TV(A_Q,H_Q) + TV(A_Q,M_Q),

    where M_Q is the deterministic quinella coarsening of M_E and M_E ranges
    over the exact marginal PriceSet induced by the shared trifecta source box.

    Absolute-value terms that enter with a positive coefficient in the active
    minimization objective need only the standard epigraph inequalities. Terms
    entering with a negative coefficient require binary sign indicators. The
    minimum and maximum models therefore use binaries for only two TV terms each.
    """
    exacta_groups = np.asarray(exacta_groups, dtype=np.int64)
    quinella_groups = np.asarray(quinella_groups, dtype=np.int64)
    h_e = np.asarray(exacta_harville, dtype=float)
    h_q = np.asarray(quinella_harville, dtype=float)
    tdim = source_set.size
    edim = exacta_actual.size
    qdim = quinella_actual.size
    if len(exacta_groups) != tdim or len(quinella_groups) != tdim:
        raise ValueError("P3 joint MILP group maps must match source dimension")
    if h_e.shape != (edim,) or h_q.shape != (qdim,):
        raise ValueError("P3 joint MILP Harville dimensions do not match targets")
    if np.any(exacta_groups < 0) or np.any(exacta_groups >= edim):
        raise ValueError("invalid exacta group map")
    if np.any(quinella_groups < 0) or np.any(quinella_groups >= qdim):
        raise ValueError("invalid quinella group map")
    if not np.isclose(h_e.sum(), 1.0) or not np.isclose(h_q.sum(), 1.0):
        raise ValueError("Harville marginals must sum to one")

    main_exacta_set = aggregate_price_set(source_set, exacta_groups, edim)
    e_to_q = _exacta_to_quinella_map(exacta_groups, quinella_groups, edim, qdim)
    q_members = [np.flatnonzero(e_to_q == qidx) for qidx in range(qdim)]

    def solve(*, maximize: bool) -> float:
        cursor = 0

        def alloc(size: int) -> slice:
            nonlocal cursor
            out = slice(cursor, cursor + size)
            cursor += size
            return out

        m_e = alloc(edim)
        sm_idx = cursor
        cursor += 1
        a_e = alloc(edim)
        se_idx = cursor
        cursor += 1
        a_q = alloc(qdim)
        sq_idx = cursor
        cursor += 1

        z_eh = alloc(edim)
        z_em = alloc(edim)
        z_qh = alloc(qdim)
        z_qm = alloc(qdim)

        exact_terms = ("eh", "qm") if maximize else ("em", "qh")
        binary_slices: dict[str, slice] = {}
        for name, size in (("eh", edim), ("em", edim), ("qh", qdim), ("qm", qdim)):
            if name in exact_terms:
                binary_slices[name] = alloc(size)

        nvar = cursor
        var_lb = np.zeros(nvar, dtype=float)
        var_ub = np.ones(nvar, dtype=float)
        var_lb[sm_idx], var_ub[sm_idx] = _scale_bounds(main_exacta_set)
        var_lb[se_idx], var_ub[se_idx] = _scale_bounds(exacta_actual)
        var_lb[sq_idx], var_ub[sq_idx] = _scale_bounds(quinella_actual)
        integrality = np.zeros(nvar, dtype=np.int8)
        for b in binary_slices.values():
            integrality[b] = 1

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        lower_rows: list[float] = []
        upper_rows: list[float] = []

        def add_row(items: list[tuple[int, float]], lower: float, upper: float) -> None:
            row = len(lower_rows)
            for col, value in items:
                if value:
                    rows.append(row)
                    cols.append(col)
                    data.append(float(value))
            lower_rows.append(float(lower))
            upper_rows.append(float(upper))

        def add_price_set(prob_slice: slice, scale_idx: int, ps: PriceSet) -> None:
            for i, (lo, hi) in enumerate(zip(ps.lower, ps.upper)):
                pidx = prob_slice.start + i
                add_row([(pidx, 1.0), (scale_idx, -float(hi))], -np.inf, 0.0)
                add_row([(pidx, -1.0), (scale_idx, float(lo))], -np.inf, 0.0)
            add_row(
                [(prob_slice.start + i, 1.0) for i in range(prob_slice.stop - prob_slice.start)],
                1.0,
                1.0,
            )

        add_price_set(m_e, sm_idx, main_exacta_set)
        add_price_set(a_e, se_idx, exacta_actual)
        add_price_set(a_q, sq_idx, quinella_actual)

        def add_abs(
            coeffs: list[tuple[int, float]],
            constant: float,
            zidx: int,
            *,
            bidx: int | None,
        ) -> None:
            add_row(coeffs + [(zidx, -1.0)], -np.inf, -constant)
            add_row(
                [(col, -value) for col, value in coeffs] + [(zidx, -1.0)],
                -np.inf,
                constant,
            )
            if bidx is None:
                return
            add_row(
                [(col, -value) for col, value in coeffs]
                + [(zidx, 1.0), (bidx, BIG_M)],
                -np.inf,
                BIG_M + constant,
            )
            add_row(
                coeffs + [(zidx, 1.0), (bidx, -BIG_M)],
                -np.inf,
                -constant,
            )

        for i in range(edim):
            ai = a_e.start + i
            mei = m_e.start + i
            add_abs(
                [(ai, 1.0)],
                -float(h_e[i]),
                z_eh.start + i,
                bidx=(binary_slices["eh"].start + i if "eh" in binary_slices else None),
            )
            add_abs(
                [(ai, 1.0), (mei, -1.0)],
                0.0,
                z_em.start + i,
                bidx=(binary_slices["em"].start + i if "em" in binary_slices else None),
            )

        for i in range(qdim):
            ai = a_q.start + i
            main_q_coeffs = [(m_e.start + int(eidx), -1.0) for eidx in q_members[i]]
            add_abs(
                [(ai, 1.0)],
                -float(h_q[i]),
                z_qh.start + i,
                bidx=(binary_slices["qh"].start + i if "qh" in binary_slices else None),
            )
            add_abs(
                [(ai, 1.0)] + main_q_coeffs,
                0.0,
                z_qm.start + i,
                bidx=(binary_slices["qm"].start + i if "qm" in binary_slices else None),
            )

        matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(lower_rows), nvar)).tocsr()
        constraints = LinearConstraint(
            matrix,
            np.asarray(lower_rows, dtype=float),
            np.asarray(upper_rows, dtype=float),
        )
        objective = np.zeros(nvar, dtype=float)
        objective[z_eh] = 0.5
        objective[z_em] = -0.5
        objective[z_qh] = -0.5
        objective[z_qm] = 0.5
        if maximize:
            objective = -objective

        result = milp(
            objective,
            integrality=integrality,
            bounds=Bounds(var_lb, var_ub),
            constraints=constraints,
            options={"time_limit": float(time_limit), "mip_rel_gap": 1e-8, "presolve": True},
        )
        if not result.success:
            direction = "maximum" if maximize else "minimum"
            raise RuntimeError(f"P3 joint {direction} MILP failed: {result.message}")
        return float(-result.fun if maximize else result.fun)

    lower = solve(maximize=False)
    upper = solve(maximize=True)
    if lower - upper > 10 * LP_TOL:
        raise RuntimeError("P3 joint MILP bounds are inverted")
    return max(-1.0, lower), min(1.0, upper)


def _representative_races(races: pd.DataFrame, full_ids: set[str]) -> list[str]:
    subset = races[races["race_id"].isin(full_ids)][["race_id", "n_valid_horses"]].copy()
    selected: list[str] = []
    for _, group in subset.groupby("n_valid_horses", sort=True):
        ids = group["race_id"].astype(str).tolist()
        selected.append(min(ids, key=lambda race_id: stable_uint(f"P3-joint|{race_id}")))
    return selected


def _conservative_race_interval(bounds: pd.DataFrame, race_id: str) -> tuple[float, float]:
    frame = bounds[bounds["race_id"].eq(race_id)]
    lower = frame.pivot(index="target_market", columns="model", values="tv_lower")
    upper = frame.pivot(index="target_market", columns="model", values="tv_upper_outer")
    lo = (
        lower.loc["exacta", "harville"]
        - upper.loc["exacta", "main"]
        - upper.loc["quinella", "harville"]
        + lower.loc["quinella", "main"]
    )
    hi = (
        upper.loc["exacta", "harville"]
        - lower.loc["exacta", "main"]
        - lower.loc["quinella", "harville"]
        + upper.loc["quinella", "main"]
    )
    return float(lo), float(hi)


def run_sharpness_diagnostic(
    data_root: Path,
    sample_csv: Path,
    bounds_csv: Path,
) -> pd.DataFrame:
    sample = read_frozen_sample(sample_csv)
    full_ids = set(common_race_ids(sample, "eligible_complete_sample"))
    races = race_metadata(data_root, full_ids)
    selected = _representative_races(races, full_ids)
    selected_set = set(selected)
    source = load_market(data_root, "trifecta", selected_set)
    win = load_market(data_root, "win", selected_set)
    exacta = load_market(data_root, "exacta", selected_set)
    quinella = load_market(data_root, "quinella", selected_set)
    bounds = pd.read_csv(bounds_csv)
    n_map = races.set_index("race_id")["n_valid_horses"].astype(int).to_dict()

    rows: list[dict[str, object]] = []
    for race_id in selected:
        n_horses = int(n_map[race_id])
        print(f"P3 joint MILP: race={race_id} field_size={n_horses}", flush=True)
        source_frame = source.get(race_id)
        win_frame = win.get(race_id)
        e_frame = exacta.get(race_id)
        q_frame = quinella.get(race_id)
        e_groups = source_group_index(source_frame, e_frame, "exacta")
        q_groups = source_group_index(source_frame, q_frame, "quinella")
        source_set = interval_for_frame(source_frame)
        e_actual = interval_for_frame(e_frame)
        q_actual = interval_for_frame(q_frame)
        h_source = harville_trifecta(source_frame, win_frame)
        h_e = aggregate_point(h_source, e_groups, len(e_frame))
        h_q = aggregate_point(h_source, q_groups, len(q_frame))
        joint_lo, joint_hi = joint_p3_extrema(
            source_set,
            e_groups,
            e_actual,
            h_e,
            q_groups,
            q_actual,
            h_q,
        )
        cons_lo, cons_hi = _conservative_race_interval(bounds, race_id)
        cons_width = cons_hi - cons_lo
        joint_width = joint_hi - joint_lo
        rows.append(
            {
                "race_id": race_id,
                "n_valid_horses": n_horses,
                "conservative_lower": cons_lo,
                "conservative_upper": cons_hi,
                "joint_milp_lower": joint_lo,
                "joint_milp_upper": joint_hi,
                "conservative_width": cons_width,
                "joint_milp_width": joint_width,
                "width_ratio_joint_to_conservative": (
                    joint_width / cons_width if cons_width > 0 else np.nan
                ),
            }
        )
        print(
            "P3 joint MILP result: "
            f"[{joint_lo:.6f}, {joint_hi:.6f}] vs conservative "
            f"[{cons_lo:.6f}, {cons_hi:.6f}]",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values(["n_valid_horses", "race_id"]).reset_index(drop=True)


def write_table(frame: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"출전두수 & 기존 보수구간 & 공유가격 MILP 구간 & 폭 비율 \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{int(row.n_valid_horses)} & "
            f"[{row.conservative_lower:.4f}, {row.conservative_upper:.4f}] & "
            f"[{row.joint_milp_lower:.4f}, {row.joint_milp_upper:.4f}] & "
            f"{row.width_ratio_joint_to_conservative:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("KRA/parsed"))
    parser.add_argument("--sample-csv", type=Path, default=Path("outputs/analysis_sample.csv"))
    parser.add_argument("--bounds-csv", type=Path, default=Path("outputs/main_metrics_bounds.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/main_order_information_joint.csv"))
    parser.add_argument("--table", type=Path, default=Path("tables/main_order_information_joint.tex"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = run_sharpness_diagnostic(args.data_root, args.sample_csv, args.bounds_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False, float_format="%.12g")
    write_table(frame, args.table)
    print(frame.to_string(index=False))
    print("PASS: P3 shared-source joint MILP sharpness diagnostic completed")


if __name__ == "__main__":
    main()
