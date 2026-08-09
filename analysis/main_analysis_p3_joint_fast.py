"""Certified shared-source P3 sharpness diagnostic with tight MILP bounds.

The co-primary Panel B P3 decision remains the pre-registered conservative
endpoint combination.  This module is a sensitivity diagnostic: it preserves
the shared trifecta-price uncertainty through the exacta marginal, exploits the
fact that quinella is a deterministic coarsening of exacta, and solves a much
tighter mixed-integer formulation.

If HiGHS reaches the time limit before proving optimality, the diagnostic does
not fail or pretend that the incumbent is sharp.  Instead it uses HiGHS' MIP
dual bound to report a certified outer bound for that direction.  Thus every
reported joint interval remains valid; rows are labelled as sharp only when both
directions are proven optimal.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    price_set_component_bounds,
    source_group_index,
)
from analysis.main_analysis_p3_joint import (
    _conservative_race_interval,
    _exacta_to_quinella_map,
    _representative_races,
)
from analysis.main_analysis_panels import interval_for_frame, load_market
from analysis.main_analysis_runner import common_race_ids, race_metadata, read_frozen_sample


@dataclass(frozen=True)
class DirectionResult:
    certified_value: float
    incumbent_value: float
    optimal: bool
    mip_gap: float
    message: str


def _scale_bounds(price_set: PriceSet) -> tuple[float, float]:
    lower_sum = float(price_set.lower.sum())
    upper_sum = float(price_set.upper.sum())
    return 1.0 / upper_sum, np.inf if lower_sum <= 0 else 1.0 / lower_sum


def _finite_float(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def joint_p3_certified(
    source_set: PriceSet,
    exacta_groups: np.ndarray,
    exacta_actual: PriceSet,
    exacta_harville: np.ndarray,
    quinella_groups: np.ndarray,
    quinella_actual: PriceSet,
    quinella_harville: np.ndarray,
    *,
    time_limit: float = 180.0,
) -> tuple[float, float, DirectionResult, DirectionResult]:
    """Return a certified joint interval and solver diagnostics for P3.

    When both directions are solved to optimality, the returned interval is the
    sharp interval for the selected race under the maintained price sets.  On a
    time limit, the appropriate MIP dual bound is used so that the interval is
    still a valid outer bound on the sharp interval.
    """
    exacta_groups = np.asarray(exacta_groups, dtype=np.int64)
    quinella_groups = np.asarray(quinella_groups, dtype=np.int64)
    h_e = np.asarray(exacta_harville, dtype=float)
    h_q = np.asarray(quinella_harville, dtype=float)
    tdim = source_set.size
    edim = exacta_actual.size
    qdim = quinella_actual.size
    if len(exacta_groups) != tdim or len(quinella_groups) != tdim:
        raise ValueError("P3 joint group maps must match source dimension")
    if h_e.shape != (edim,) or h_q.shape != (qdim,):
        raise ValueError("P3 joint Harville dimensions do not match targets")
    if not np.isclose(h_e.sum(), 1.0) or not np.isclose(h_q.sum(), 1.0):
        raise ValueError("P3 joint Harville marginals must sum to one")

    main_e_set = aggregate_price_set(source_set, exacta_groups, edim)
    main_q_set = aggregate_price_set(source_set, quinella_groups, qdim)
    e_to_q = _exacta_to_quinella_map(exacta_groups, quinella_groups, edim, qdim)
    q_members = [np.flatnonzero(e_to_q == qidx) for qidx in range(qdim)]

    me_lo, me_hi = price_set_component_bounds(main_e_set)
    mq_lo, mq_hi = price_set_component_bounds(main_q_set)
    ae_lo, ae_hi = price_set_component_bounds(exacta_actual)
    aq_lo, aq_hi = price_set_component_bounds(quinella_actual)

    d_bounds = {
        "eh": (ae_lo - h_e, ae_hi - h_e),
        "em": (ae_lo - me_hi, ae_hi - me_lo),
        "qh": (aq_lo - h_q, aq_hi - h_q),
        "qm": (aq_lo - mq_hi, aq_hi - mq_lo),
    }

    def solve(*, maximize: bool) -> DirectionResult:
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

        # In the minimization presented to HiGHS, only absolute-value terms with
        # a negative objective coefficient require exact sign modelling.
        exact_terms = {"eh", "qm"} if maximize else {"em", "qh"}
        binary_index: dict[tuple[str, int], int] = {}
        for name, size in (("eh", edim), ("em", edim), ("qh", qdim), ("qm", qdim)):
            if name not in exact_terms:
                continue
            lo, hi = d_bounds[name]
            for i in range(size):
                if float(lo[i]) < -LP_TOL and float(hi[i]) > LP_TOL:
                    binary_index[(name, i)] = cursor
                    cursor += 1

        nvar = cursor
        var_lb = np.zeros(nvar, dtype=float)
        var_ub = np.ones(nvar, dtype=float)
        var_lb[m_e], var_ub[m_e] = me_lo, me_hi
        var_lb[a_e], var_ub[a_e] = ae_lo, ae_hi
        var_lb[a_q], var_ub[a_q] = aq_lo, aq_hi
        var_lb[sm_idx], var_ub[sm_idx] = _scale_bounds(main_e_set)
        var_lb[se_idx], var_ub[se_idx] = _scale_bounds(exacta_actual)
        var_lb[sq_idx], var_ub[sq_idx] = _scale_bounds(quinella_actual)

        for name, zslice in (("eh", z_eh), ("em", z_em), ("qh", z_qh), ("qm", z_qm)):
            lo, hi = d_bounds[name]
            var_ub[zslice] = np.maximum(np.abs(lo), np.abs(hi))

        integrality = np.zeros(nvar, dtype=np.int8)
        for idx in binary_index.values():
            integrality[idx] = 1

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

        add_price_set(m_e, sm_idx, main_e_set)
        add_price_set(a_e, se_idx, exacta_actual)
        add_price_set(a_q, sq_idx, quinella_actual)

        def add_abs(
            name: str,
            i: int,
            coeffs: list[tuple[int, float]],
            constant: float,
            zidx: int,
            *,
            exact: bool,
        ) -> None:
            dlo = float(d_bounds[name][0][i])
            dhi = float(d_bounds[name][1][i])
            # Epigraph: z >= d and z >= -d.
            add_row(coeffs + [(zidx, -1.0)], -np.inf, -constant)
            add_row(
                [(col, -value) for col, value in coeffs] + [(zidx, -1.0)],
                -np.inf,
                constant,
            )
            if not exact:
                return
            if dlo >= -LP_TOL:
                # d is nonnegative: z=d exactly, no binary variable needed.
                add_row(
                    [(zidx, 1.0)] + [(col, -value) for col, value in coeffs],
                    constant,
                    constant,
                )
                return
            if dhi <= LP_TOL:
                # d is nonpositive: z=-d exactly, no binary variable needed.
                add_row(
                    [(zidx, 1.0)] + coeffs,
                    -constant,
                    -constant,
                )
                return

            bidx = binary_index[(name, i)]
            # b=1 selects d>=0; b=0 selects d<=0.  The component-specific
            # bounds dlo/dhi replace the old global M=2 relaxation.
            add_row(coeffs + [(bidx, -dhi)], -np.inf, -constant)
            add_row(
                [(col, -value) for col, value in coeffs] + [(bidx, -dlo)],
                -np.inf,
                constant - dlo,
            )
            add_row(
                [(zidx, 1.0)]
                + [(col, -value) for col, value in coeffs]
                + [(bidx, -2.0 * dlo)],
                -np.inf,
                constant - 2.0 * dlo,
            )
            add_row(
                [(zidx, 1.0)] + coeffs + [(bidx, -2.0 * dhi)],
                -np.inf,
                -constant,
            )

        for i in range(edim):
            ai = a_e.start + i
            mei = m_e.start + i
            add_abs(
                "eh", i, [(ai, 1.0)], -float(h_e[i]), z_eh.start + i,
                exact="eh" in exact_terms,
            )
            add_abs(
                "em", i, [(ai, 1.0), (mei, -1.0)], 0.0, z_em.start + i,
                exact="em" in exact_terms,
            )

        for i in range(qdim):
            ai = a_q.start + i
            main_q_coeffs = [(m_e.start + int(eidx), -1.0) for eidx in q_members[i]]
            add_abs(
                "qh", i, [(ai, 1.0)], -float(h_q[i]), z_qh.start + i,
                exact="qh" in exact_terms,
            )
            add_abs(
                "qm", i, [(ai, 1.0)] + main_q_coeffs, 0.0, z_qm.start + i,
                exact="qm" in exact_terms,
            )

        matrix = sparse.coo_matrix(
            (data, (rows, cols)), shape=(len(lower_rows), nvar)
        ).tocsr()
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
            options={
                "time_limit": float(time_limit),
                "mip_rel_gap": 1e-8,
                "presolve": True,
            },
        )
        fun = _finite_float(getattr(result, "fun", np.nan), np.nan)
        dual = _finite_float(getattr(result, "mip_dual_bound", np.nan), np.nan)
        gap = _finite_float(getattr(result, "mip_gap", np.nan), np.nan)
        if result.success:
            value = -fun if maximize else fun
            return DirectionResult(value, value, True, gap, str(result.message))

        # A time-limited MIP can still provide a mathematically useful certified
        # bound.  For min f, dual<=min f.  For max f we solved min(-f), so
        # -dual>=max f.  The incumbent is recorded separately and never promoted
        # to a certified endpoint.
        if np.isfinite(dual):
            certified = -dual if maximize else dual
        else:
            certified = 1.0 if maximize else -1.0
        incumbent = (-fun if maximize else fun) if np.isfinite(fun) else np.nan
        return DirectionResult(
            max(-1.0, min(1.0, certified)),
            incumbent,
            False,
            gap,
            str(result.message),
        )

    minimum = solve(maximize=False)
    maximum = solve(maximize=True)
    lower = max(-1.0, minimum.certified_value)
    upper = min(1.0, maximum.certified_value)
    if lower - upper > 10 * LP_TOL:
        raise RuntimeError("P3 certified joint MILP bounds are inverted")
    return lower, upper, minimum, maximum


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
        print(f"P3 tight joint MILP: race={race_id} field_size={n_horses}", flush=True)
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
        joint_lo, joint_hi, min_result, max_result = joint_p3_certified(
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
                "joint_min_optimal": min_result.optimal,
                "joint_max_optimal": max_result.optimal,
                "joint_min_incumbent": min_result.incumbent_value,
                "joint_max_incumbent": max_result.incumbent_value,
                "joint_min_mip_gap": min_result.mip_gap,
                "joint_max_mip_gap": max_result.mip_gap,
            }
        )
        status = "sharp" if min_result.optimal and max_result.optimal else "certified outer"
        print(
            "P3 tight joint result: "
            f"[{joint_lo:.6f}, {joint_hi:.6f}] ({status}) vs conservative "
            f"[{cons_lo:.6f}, {cons_hi:.6f}]",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values(["n_valid_horses", "race_id"]).reset_index(drop=True)


def write_table(frame: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{tabular}{rrrrl}",
        r"\toprule",
        r"출전두수 & 기존 보수구간 & 공유가격 MILP 인증구간 & 폭 비율 & 해 상태 \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        status = "sharp" if row.joint_min_optimal and row.joint_max_optimal else "certified"
        lines.append(
            f"{int(row.n_valid_horses)} & "
            f"[{row.conservative_lower:.4f}, {row.conservative_upper:.4f}] & "
            f"[{row.joint_milp_lower:.4f}, {row.joint_milp_upper:.4f}] & "
            f"{row.width_ratio_joint_to_conservative:.3f} & {status} \\\\"
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
    print("PASS: P3 tight shared-source certified MILP diagnostic completed")


if __name__ == "__main__":
    main()
