#!/usr/bin/env python3
"""Independent re-verification for the Cultural Industry journal version.

This module intentionally re-computes the clean-sample cross-pool TV statistics
without calling the main-analysis pricing functions.  It also freezes race-day
cluster bootstrap intervals, the 2018-07-01 >9999.9 diagnostic, turnover-based
mechanical noise benchmarks, and SHA-256 hashes of every parsed parquet input.

The raw KRA JSON archive is not distributed in this repository, so this audit is
an independent re-computation from the versioned parsed parquet snapshot.  The
input-file hash inventory makes that exact snapshot recoverable later.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq


DATA_ROOT = Path("KRA/parsed")
OUTPUT_DIR = Path("outputs")
DOC_PATH = Path("submission/cultural-industry/REVERIFICATION_2026-08-16.md")
START_DATE = pd.Timestamp("2016-06-10")
END_DATE = pd.Timestamp("2025-12-31")
UNAVAILABLE_YEARS = frozenset({2020, 2021})
DISPLAY_CAP = 9999.9
CAP_ATOL = 1e-9
NOMINAL_WAGER_UNIT_WON = 100
BETTING_LIMIT_WON_PER_PERSON_RACE = 100_000
BOOTSTRAP_REPS = 4_999
BOOTSTRAP_SEED = 20260816
TURNOVER_MC_DRAWS = 256
TURNOVER_MC_SEED = 20260816
TARGETS = ("win", "exacta", "quinella", "trio")
COMPOUND_TARGETS = ("exacta", "quinella", "trio")
MARKETS = (*TARGETS, "trifecta")
KEYS = {
    "win": ("horse_no",),
    "exacta": ("first_no", "second_no"),
    "quinella": ("horse_a", "horse_b"),
    "trio": ("horse_a", "horse_b", "horse_c"),
    "trifecta": ("first_no", "second_no", "third_no"),
}


def stable_seed(label: str, base: int) -> int:
    digest = hashlib.sha256(f"{base}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def parquet_paths(kind: str) -> list[Path]:
    if kind in MARKETS or kind in {"place", "quinella_place"}:
        pattern = f"*/market={kind}/year=*/month=*/part-*.parquet"
    else:
        pattern = f"*/{kind}/year=*/month=*/part-*.parquet"
    paths = sorted(DATA_ROOT.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet files for {kind}: {pattern}")
    return paths


def read_parquets(kind: str, columns: Iterable[str] | None = None) -> pd.DataFrame:
    return pq.read_table(parquet_paths(kind), columns=columns).to_pandas()


def expected_rows(market: str, n: pd.Series) -> pd.Series:
    if market == "win":
        return n
    if market == "exacta":
        return n * (n - 1)
    if market == "quinella":
        return n * (n - 1) // 2
    if market == "trio":
        return n * (n - 1) * (n - 2) // 6
    if market == "trifecta":
        return n * (n - 1) * (n - 2)
    raise ValueError(market)


def canonical_cap(odds: pd.Series) -> pd.Series:
    values = pd.to_numeric(odds, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.isclose(values, DISPLAY_CAP, rtol=0.0, atol=CAP_ATOL), index=odds.index)


def load_races() -> pd.DataFrame:
    races = read_parquets(
        "valid_horses",
        columns=["race_id", "race_date", "meet", "race_no", "n_valid_horses", "valid_horses"],
    ).copy()
    if races["race_id"].duplicated().any():
        raise AssertionError("duplicate race_id in valid_horses")
    dates = pd.to_datetime(races["race_date"], errors="raise")
    races["race_date"] = dates.dt.strftime("%Y-%m-%d")
    races["in_scope"] = (
        dates.ge(START_DATE)
        & dates.le(END_DATE)
        & ~dates.dt.year.isin(UNAVAILABLE_YEARS)
    )
    return races.sort_values("race_id").reset_index(drop=True)


def load_status() -> pd.DataFrame:
    status = read_parquets("market_status", columns=None).copy()
    if status.duplicated(["race_id", "market"]).any():
        raise AssertionError("duplicate race_id-market in market_status")
    if "turnover_won" in status.columns:
        status["turnover_won"] = pd.to_numeric(status["turnover_won"], errors="coerce")
    return status


def market_quality(market: str, races: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    cols = ["race_id", *KEYS[market], "odds", "is_capped_odds"]
    if market != "win":
        cols.append("is_cancel")
    frame = read_parquets(market, columns=cols).copy()
    frame["race_id"] = frame["race_id"].astype(str)
    odds = pd.to_numeric(frame["odds"], errors="coerce")
    frame["_invalid_odds"] = (~np.isfinite(odds.to_numpy(dtype=float))) | odds.le(0)
    frame["_canonical_cap"] = canonical_cap(odds).to_numpy()
    frame["_gt_9999_9"] = odds.gt(DISPLAY_CAP)
    stored = frame["is_capped_odds"].fillna(False).astype(bool)
    frame["_stored_cap"] = stored
    frame["_stored_true_gt_cap"] = stored & frame["_gt_9999_9"]
    frame["_cap_mismatch_non_gt"] = stored.ne(frame["_canonical_cap"]) & ~frame["_gt_9999_9"]
    frame["_duplicate"] = frame.duplicated(["race_id", *KEYS[market]], keep=False)
    if market == "win":
        frame["_cancel"] = False
    else:
        frame["_cancel"] = frame["is_cancel"].fillna(True).astype(bool)

    grouped = frame.groupby("race_id", sort=False).agg(
        observed_rows=("race_id", "size"),
        duplicate_rows=("_duplicate", "sum"),
        invalid_odds_rows=("_invalid_odds", "sum"),
        canonical_cap_rows=("_canonical_cap", "sum"),
        odds_gt_9999_9_rows=("_gt_9999_9", "sum"),
        stored_cap_rows=("_stored_cap", "sum"),
        stored_true_gt_cap_rows=("_stored_true_gt_cap", "sum"),
        cap_mismatch_non_gt_rows=("_cap_mismatch_non_gt", "sum"),
        cancelled_rows=("_cancel", "sum"),
    )
    base = races[["race_id", "n_valid_horses", "in_scope"]].copy()
    base["race_id"] = base["race_id"].astype(str)
    out = base.set_index("race_id").join(grouped).fillna(0)
    out["expected_rows"] = expected_rows(market, out["n_valid_horses"].astype(int))

    status_m = status[status["market"].eq(market)].copy()
    status_m["race_id"] = status_m["race_id"].astype(str)
    status_m = status_m.set_index("race_id")
    if "n_rows" in status_m.columns:
        out["status_n_rows"] = status_m["n_rows"]
    else:
        out["status_n_rows"] = np.nan
    out["status_name"] = status_m["status"] if "status" in status_m.columns else np.nan
    out["status_reason"] = status_m["status_reason"] if "status_reason" in status_m.columns else np.nan
    out["status_cancelled"] = status_m["is_cancelled"] if "is_cancelled" in status_m.columns else False
    status_ok = out["status_name"].eq("ok")
    if out["status_n_rows"].notna().any():
        status_ok &= out["status_n_rows"].fillna(-1).astype(int).eq(out["observed_rows"].astype(int))
    if "status_reason" in status_m.columns:
        status_ok &= out["status_reason"].eq("parsed_rows_present")
    status_ok &= ~out["status_cancelled"].fillna(True).astype(bool)
    out["structurally_valid"] = (
        out["in_scope"].astype(bool)
        & out["observed_rows"].astype(int).eq(out["expected_rows"].astype(int))
        & out["duplicate_rows"].astype(int).eq(0)
        & out["invalid_odds_rows"].astype(int).eq(0)
        & out["cap_mismatch_non_gt_rows"].astype(int).eq(0)
        & out["cancelled_rows"].astype(int).eq(0)
        & status_ok
    )
    out["uncapped"] = out["canonical_cap_rows"].astype(int).eq(0)
    out["market"] = market
    return out.reset_index()


def normalized_inverse_odds(frame: pd.DataFrame) -> pd.Series:
    inv = 1.0 / pd.to_numeric(frame["odds"], errors="raise").astype(float)
    total = float(inv.sum())
    if not np.isfinite(total) or total <= 0:
        raise AssertionError("invalid inverse-odds total")
    return inv / total


def indexed_price(frame: pd.DataFrame, market: str) -> pd.Series:
    local = frame.copy()
    local["prob"] = normalized_inverse_odds(local).to_numpy()
    result = local.set_index(list(KEYS[market]))["prob"].sort_index()
    if result.index.has_duplicates:
        raise AssertionError(f"duplicate observed support: {market}")
    return result


def marginalize_source(source: pd.DataFrame, probability: np.ndarray, target: str) -> pd.Series:
    local = source[["first_no", "second_no", "third_no"]].copy()
    local["prob"] = probability
    if target == "win":
        result = local.groupby("first_no", sort=True)["prob"].sum()
        result.index.name = "horse_no"
        return result.sort_index()
    if target == "exacta":
        return local.groupby(["first_no", "second_no"], sort=True)["prob"].sum().sort_index()
    if target == "quinella":
        a = np.minimum(local["first_no"].to_numpy(int), local["second_no"].to_numpy(int))
        b = np.maximum(local["first_no"].to_numpy(int), local["second_no"].to_numpy(int))
        local["horse_a"] = a
        local["horse_b"] = b
        return local.groupby(["horse_a", "horse_b"], sort=True)["prob"].sum().sort_index()
    if target == "trio":
        ordered = np.sort(local[["first_no", "second_no", "third_no"]].to_numpy(int), axis=1)
        local[["horse_a", "horse_b", "horse_c"]] = ordered
        return local.groupby(["horse_a", "horse_b", "horse_c"], sort=True)["prob"].sum().sort_index()
    raise ValueError(target)


def harville_on_trifecta_support(source: pd.DataFrame, win: pd.DataFrame) -> np.ndarray:
    p_series = indexed_price(win, "win")
    p = {int(k): float(v) for k, v in p_series.items()}
    i = source["first_no"].to_numpy(int)
    j = source["second_no"].to_numpy(int)
    k = source["third_no"].to_numpy(int)
    pi = np.fromiter((p[int(x)] for x in i), dtype=float, count=len(i))
    pj = np.fromiter((p[int(x)] for x in j), dtype=float, count=len(j))
    pk = np.fromiter((p[int(x)] for x in k), dtype=float, count=len(k))
    d2 = 1.0 - pi
    d3 = 1.0 - pi - pj
    if np.any(d2 <= 0) or np.any(d3 <= 0):
        raise AssertionError("invalid Harville denominator")
    h = pi * (pj / d2) * (pk / d3)
    total = float(h.sum())
    if not np.isfinite(total) or abs(total - 1.0) > 2e-10:
        raise AssertionError(f"Harville trifecta mass does not sum to one: {total}")
    return h / total


def tv_distance(a: pd.Series, b: pd.Series) -> float:
    if not a.index.equals(b.index):
        missing_a = b.index.difference(a.index)
        missing_b = a.index.difference(b.index)
        raise AssertionError(f"support mismatch: missing_a={len(missing_a)}, missing_b={len(missing_b)}")
    return float(0.5 * np.abs(a.to_numpy(float) - b.to_numpy(float)).sum())


def race_bootstrap_ci(values: np.ndarray, label: str) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed(label, BOOTSTRAP_SEED))
    n = len(values)
    estimates = np.empty(BOOTSTRAP_REPS, dtype=float)
    for r in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, size=n)
        estimates[r] = float(np.median(values[idx]))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def date_cluster_bootstrap_ci(frame: pd.DataFrame, label: str) -> tuple[float, float, int]:
    groups = [g["tv"].to_numpy(float) for _, g in frame.groupby("race_date", sort=True)]
    rng = np.random.default_rng(stable_seed(label, BOOTSTRAP_SEED + 1))
    g = len(groups)
    estimates = np.empty(BOOTSTRAP_REPS, dtype=float)
    for r in range(BOOTSTRAP_REPS):
        sampled = rng.integers(0, g, size=g)
        values = np.concatenate([groups[i] for i in sampled])
        estimates[r] = float(np.median(values))
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        g,
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hash_inventory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    kinds = ["valid_horses", "market_status", *MARKETS]
    for kind in kinds:
        for path in parquet_paths(kind):
            rows.append(
                {
                    "kind": kind,
                    "path": path.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return pd.DataFrame(rows).sort_values(["kind", "path"]).reset_index(drop=True)


def overcap_check(races: pd.DataFrame) -> pd.DataFrame:
    target_date = "2018-07-01"
    rows: list[dict[str, object]] = []
    date_ids = set(races.loc[races["race_date"].eq(target_date), "race_id"].astype(str))
    for market in MARKETS:
        frame = read_parquets(market, columns=["race_id", "odds", "is_capped_odds"])
        frame["race_id"] = frame["race_id"].astype(str)
        frame = frame[frame["race_id"].isin(date_ids)].copy()
        odds = pd.to_numeric(frame["odds"], errors="coerce")
        stored = frame["is_capped_odds"].fillna(False).astype(bool)
        rows.append(
            {
                "market": market,
                "date_races": frame["race_id"].nunique(),
                "rows": len(frame),
                "odds_eq_9999_9": int(np.isclose(odds.to_numpy(float), DISPLAY_CAP, rtol=0.0, atol=CAP_ATOL).sum()),
                "odds_gt_9999_9": int(odds.gt(DISPLAY_CAP).sum()),
                "stored_capped_rows": int(stored.sum()),
                "stored_capped_and_gt_9999_9": int((stored & odds.gt(DISPLAY_CAP)).sum()),
                "max_odds": float(odds.max()) if len(odds) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_clean_ids(races: pd.DataFrame, status: pd.DataFrame) -> tuple[dict[str, set[str]], pd.DataFrame]:
    qualities = {m: market_quality(m, races, status) for m in MARKETS}
    source = qualities["trifecta"].set_index("race_id")
    clean: dict[str, set[str]] = {}
    audit_rows: list[dict[str, object]] = []
    in_scope_ids = set(races.loc[races["in_scope"], "race_id"].astype(str))
    for market in TARGETS:
        target = qualities[market].set_index("race_id")
        ok = (
            source["structurally_valid"].astype(bool)
            & target["structurally_valid"].astype(bool)
            & source["uncapped"].astype(bool)
            & target["uncapped"].astype(bool)
        )
        ids = set(ok[ok].index.astype(str)) & in_scope_ids
        clean[market] = ids
        audit_rows.append(
            {
                "target_market": market,
                "candidate_in_scope_races": len(in_scope_ids),
                "clean_races": len(ids),
                "trifecta_capped_in_scope": int((source.loc[list(in_scope_ids), "canonical_cap_rows"] > 0).sum()),
                "target_capped_in_scope": int((target.loc[list(in_scope_ids), "canonical_cap_rows"] > 0).sum()),
            }
        )
    first = clean[TARGETS[0]]
    if any(clean[m] != first for m in TARGETS[1:]):
        raise AssertionError("four target clean race-id sets are not identical")
    return clean, pd.DataFrame(audit_rows)


def load_clean_frames(common_ids: set[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for market in TARGETS:
        cols = ["race_id", *KEYS[market], "odds"]
        frame = read_parquets(market, columns=cols)
        frame["race_id"] = frame["race_id"].astype(str)
        frames[market] = frame[frame["race_id"].isin(common_ids)].copy()
    source = read_parquets(
        "trifecta",
        columns=["race_id", "first_no", "second_no", "third_no", "odds"],
    )
    source["race_id"] = source["race_id"].astype(str)
    frames["trifecta"] = source[source["race_id"].isin(common_ids)].copy()
    return frames


def compute_per_race_tv(races: pd.DataFrame, common_ids: set[str]) -> pd.DataFrame:
    frames = load_clean_frames(common_ids)
    groups = {
        market: {str(k): g for k, g in frame.groupby("race_id", sort=False)}
        for market, frame in frames.items()
    }
    date_map = races.set_index(races["race_id"].astype(str))["race_date"].to_dict()
    rows: list[dict[str, object]] = []
    for race_id in sorted(common_ids):
        source = groups["trifecta"][race_id]
        main_prob = normalized_inverse_odds(source).to_numpy(float)
        h_prob = harville_on_trifecta_support(source, groups["win"][race_id])
        for target in TARGETS:
            observed = indexed_price(groups[target][race_id], target)
            main = marginalize_source(source, main_prob, target)
            harville = marginalize_source(source, h_prob, target)
            if abs(float(main.sum()) - 1.0) > 2e-12 or abs(float(harville.sum()) - 1.0) > 2e-12:
                raise AssertionError("marginal probability does not sum to one")
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": date_map[race_id],
                    "target_market": target,
                    "model": "main",
                    "tv": tv_distance(observed, main),
                }
            )
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": date_map[race_id],
                    "target_market": target,
                    "model": "harville",
                    "tv": tv_distance(observed, harville),
                }
            )
    return pd.DataFrame(rows)


def summarize_tv(per_race: pd.DataFrame) -> pd.DataFrame:
    tracked_path = OUTPUT_DIR / "main_panel_a_summary.csv"
    tracked = pd.read_csv(tracked_path) if tracked_path.exists() else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (target, model), frame in per_race.groupby(["target_market", "model"], sort=True):
        values = frame["tv"].to_numpy(float)
        race_low, race_high = race_bootstrap_ci(values, f"{target}:{model}:race")
        cluster_low, cluster_high, n_dates = date_cluster_bootstrap_ci(frame, f"{target}:{model}:date")
        row: dict[str, object] = {
            "target_market": target,
            "model": model,
            "n_races": len(frame),
            "n_race_dates": n_dates,
            "median_tv": float(np.median(values)),
            "race_boot_ci_low": race_low,
            "race_boot_ci_high": race_high,
            "race_day_cluster_ci_low": cluster_low,
            "race_day_cluster_ci_high": cluster_high,
            "bootstrap_reps": BOOTSTRAP_REPS,
        }
        if not tracked.empty:
            hit = tracked[(tracked["target_market"].eq(target)) & (tracked["model"].eq(model))]
            if len(hit) == 1:
                row["tracked_n_races"] = int(hit.iloc[0]["n_races"])
                row["tracked_median_tv"] = float(hit.iloc[0]["median_tv"])
                row["tracked_abs_diff"] = abs(row["median_tv"] - row["tracked_median_tv"])
                row["tracked_matches"] = bool(
                    row["n_races"] == row["tracked_n_races"]
                    and row["tracked_abs_diff"] <= 1e-10
                )
        rows.append(row)
    out = pd.DataFrame(rows)
    if "tracked_matches" in out.columns and not out["tracked_matches"].fillna(False).all():
        bad = out.loc[~out["tracked_matches"].fillna(False), ["target_market", "model", "tracked_abs_diff"]]
        raise AssertionError(f"independent TV recomputation disagrees with tracked output:\n{bad}")
    return out


def turnover_betting_limit_summary(status: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    if "turnover_won" not in status.columns:
        return pd.DataFrame([{"available": False}])
    ids = set(races.loc[races["in_scope"], "race_id"].astype(str))
    local = status[status["race_id"].astype(str).isin(ids)].copy()
    local["turnover_won"] = pd.to_numeric(local["turnover_won"], errors="coerce")
    local = local[local["turnover_won"].notna() & local["turnover_won"].gt(0)]
    grid_violations = int((local["turnover_won"] % NOMINAL_WAGER_UNIT_WON != 0).sum())
    race_total = local.groupby("race_id")["turnover_won"].sum()
    minimum_bettors = np.ceil(race_total / BETTING_LIMIT_WON_PER_PERSON_RACE)
    return pd.DataFrame(
        [
            {
                "available": True,
                "n_races_with_positive_total_turnover": int(len(race_total)),
                "turnover_rows_not_on_100_won_grid": grid_violations,
                "median_total_turnover_all_pools_won": float(race_total.median()),
                "p05_total_turnover_all_pools_won": float(race_total.quantile(0.05)),
                "p95_total_turnover_all_pools_won": float(race_total.quantile(0.95)),
                "median_minimum_bettor_equivalents_at_100k_cap": float(np.median(minimum_bettors)),
                "p05_minimum_bettor_equivalents_at_100k_cap": float(np.quantile(minimum_bettors, 0.05)),
                "p95_minimum_bettor_equivalents_at_100k_cap": float(np.quantile(minimum_bettors, 0.95)),
                "betting_limit_won_per_person_race": BETTING_LIMIT_WON_PER_PERSON_RACE,
            }
        ]
    )


def turnover_matched_null(
    status: pd.DataFrame,
    common_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "turnover_won" not in status.columns:
        return pd.DataFrame(), pd.DataFrame()
    status2 = status.copy()
    status2["race_id"] = status2["race_id"].astype(str)
    status2["turnover_won"] = pd.to_numeric(status2["turnover_won"], errors="coerce")
    wide = status2[status2["race_id"].isin(common_ids)].pivot(index="race_id", columns="market", values="turnover_won")
    frames = load_clean_frames(common_ids)
    source_groups = {str(k): g for k, g in frames["trifecta"].groupby("race_id", sort=False)}
    target_groups = {
        m: {str(k): g for k, g in frames[m].groupby("race_id", sort=False)} for m in COMPOUND_TARGETS
    }
    race_rows: list[dict[str, object]] = []
    for target in COMPOUND_TARGETS:
        for race_id in sorted(common_ids):
            if target not in wide.columns or "trifecta" not in wide.columns or race_id not in wide.index:
                continue
            t_turn = wide.at[race_id, target]
            s_turn = wide.at[race_id, "trifecta"]
            if pd.isna(t_turn) or pd.isna(s_turn) or t_turn <= 0 or s_turn <= 0:
                continue
            if int(t_turn) % NOMINAL_WAGER_UNIT_WON != 0 or int(s_turn) % NOMINAL_WAGER_UNIT_WON != 0:
                continue
            source = source_groups[race_id]
            p = marginalize_source(source, normalized_inverse_odds(source).to_numpy(float), target)
            observed = indexed_price(target_groups[target][race_id], target)
            obs_tv = tv_distance(observed, p)
            pvec = p.to_numpy(float)
            n_target = int(t_turn) // NOMINAL_WAGER_UNIT_WON
            n_source = int(s_turn) // NOMINAL_WAGER_UNIT_WON
            variance = pvec * (1.0 - pvec) * (1.0 / n_target + 1.0 / n_source)
            analytic_expected = float(0.5 * math.sqrt(2.0 / math.pi) * np.sqrt(variance).sum())
            rng = np.random.default_rng(stable_seed(f"{target}:{race_id}", TURNOVER_MC_SEED))
            a = rng.multinomial(n_target, pvec, size=TURNOVER_MC_DRAWS) / n_target
            b = rng.multinomial(n_source, pvec, size=TURNOVER_MC_DRAWS) / n_source
            tvs = 0.5 * np.abs(a - b).sum(axis=1)
            race_rows.append(
                {
                    "race_id": race_id,
                    "target_market": target,
                    "target_turnover_won": int(t_turn),
                    "trifecta_turnover_won": int(s_turn),
                    "target_nominal_100w_units": n_target,
                    "trifecta_nominal_100w_units": n_source,
                    "observed_tv": obs_tv,
                    "analytic_expected_tv": analytic_expected,
                    "mc_median_tv": float(np.median(tvs)),
                    "mc_p95_tv": float(np.quantile(tvs, 0.95)),
                    "observed_gt_mc_p95": bool(obs_tv > float(np.quantile(tvs, 0.95))),
                    "mc_draws_per_race": TURNOVER_MC_DRAWS,
                }
            )
    race_frame = pd.DataFrame(race_rows)
    summary_rows: list[dict[str, object]] = []
    if not race_frame.empty:
        for target, frame in race_frame.groupby("target_market", sort=True):
            obs = float(frame["observed_tv"].median())
            mc = float(frame["mc_median_tv"].median())
            summary_rows.append(
                {
                    "target_market": target,
                    "n_turnover_complete_clean_races": len(frame),
                    "mc_draws_per_race": TURNOVER_MC_DRAWS,
                    "nominal_wager_unit_won": NOMINAL_WAGER_UNIT_WON,
                    "median_target_turnover_won": float(frame["target_turnover_won"].median()),
                    "median_trifecta_turnover_won": float(frame["trifecta_turnover_won"].median()),
                    "median_target_nominal_100w_units": float(frame["target_nominal_100w_units"].median()),
                    "median_trifecta_nominal_100w_units": float(frame["trifecta_nominal_100w_units"].median()),
                    "observed_median_tv": obs,
                    "median_analytic_expected_tv": float(frame["analytic_expected_tv"].median()),
                    "median_mc_median_tv": mc,
                    "median_mc_p95_tv": float(frame["mc_p95_tv"].median()),
                    "observed_to_mc_median_ratio": obs / mc,
                    "share_observed_gt_race_mc_p95": float(frame["observed_gt_mc_p95"].mean()),
                }
            )
    return race_frame, pd.DataFrame(summary_rows)


def write_doc(
    sample_audit: pd.DataFrame,
    tv_summary: pd.DataFrame,
    overcap: pd.DataFrame,
    limit_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    input_hashes: pd.DataFrame,
) -> None:
    n_candidates = int(sample_audit["candidate_in_scope_races"].iloc[0])
    n_clean = int(sample_audit["clean_races"].iloc[0])
    cap_count = int(sample_audit["trifecta_capped_in_scope"].iloc[0])
    compounds = tv_summary[tv_summary["target_market"].isin(COMPOUND_TARGETS)].copy()
    lines = [
        "# 문화산업연구 원고 재검산 기록 — 2026-08-16",
        "",
        "## 목적과 원칙",
        "",
        "이 문서는 제출용 원고의 핵심 숫자를 나중에 다시 검산할 수 있도록 계산 과정,",
        "입력 데이터 스냅샷, 난수시드와 반복 수를 함께 고정한다. 주분석 함수의 결과를",
        "그대로 복사하지 않고 `analysis/cultural_industry_reverification.py`에서 역배당 정규화,",
        "삼쌍승 주변화, Harville, TV를 별도로 다시 구현해 계산했다.",
        "",
        "원시 KRA JSON gzip 파일은 이 저장소에 재배포되지 않으므로 이번 재검산의 최하단 입력은",
        "버전관리된 `KRA/parsed/` parquet이다. 사용한 모든 parquet의 SHA-256은",
        "`outputs/cultural_industry_reverification_input_files.csv`에 기록했다.",
        "",
        "## 1. 표본 재확인",
        "",
        f"- 분석기간 내 이용 가능한 경주: **{n_candidates:,}개**",
        f"- 삼쌍승 9,999.9 표시상한 포함 경주: **{cap_count:,}개**",
        f"- 표시상한 없는 공통 점표본: **{n_clean:,}개**",
        "- 네 목표 승식의 clean race_id 집합은 서로 완전히 동일함을 다시 확인했다.",
        "",
        "중요: 과거 19,284개/3,321개 표본은 2018-07-01의 17개 경주를 제외했기 때문에 나온 값이다.",
        "재검산에서는 9,999.9 **그 자체만** 검열 표시값으로 취급하고, 역사자료에 실제로 존재하는",
        "9,999.9 초과 게시배당은 점관측으로 남긴다. 따라서 현재 원고의 19,301개/3,338개가",
        "버전관리된 parsed data와 현재의 표시상한 정의에 일치한다.",
        "",
        "2018-07-01 세부 확인은 `outputs/cultural_industry_2018_overcap_check.csv`에 남겼다.",
        "",
        "## 2. 핵심 TV의 독립 재계산",
        "",
        "TV는 경주별로 `0.5 * sum(abs(observed - predicted))`로 계산했다. 삼쌍승 가격은",
        "각 경주에서 `1/odds`를 합 1로 정규화한 뒤 목표 사건별로 합산했다. Harville은 단승식의",
        "정규화 역배당 확률을 이용해 `(i,j,k)` 순서확률을",
        "`p_i * p_j/(1-p_i) * p_k/(1-p_i-p_j)`로 별도 재구성했다.",
        "",
        "|승식|모형|경주수|TV 중앙값|경주 bootstrap 95% CI|경주일 군집 bootstrap 95% CI|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    label = {"exacta": "쌍승", "quinella": "복승", "trio": "삼복승", "win": "단승"}
    for _, row in compounds.sort_values(["target_market", "model"]).iterrows():
        lines.append(
            f"|{label[row['target_market']]}|{row['model']}|{int(row['n_races']):,}|"
            f"{row['median_tv']:.6f}|[{row['race_boot_ci_low']:.6f}, {row['race_boot_ci_high']:.6f}]|"
            f"[{row['race_day_cluster_ci_low']:.6f}, {row['race_day_cluster_ci_high']:.6f}]|"
        )
    lines += [
        "",
        f"bootstrap 반복 수는 **{BOOTSTRAP_REPS:,}회**이며 고정 기준시드는 `{BOOTSTRAP_SEED}`이다.",
        "경주 bootstrap과 경주일 군집 bootstrap을 모두 남겼고, 후자는 같은 날짜의 여러 경주를",
        "한 군집으로 묶어 날짜를 복원추출한다. `tracked_matches=True`가 전 행에서 확인되어야 하며,",
        "이는 독립 재계산 중앙값이 기존 `outputs/main_panel_a_summary.csv`와 1e-10 이내로 일치함을 뜻한다.",
        "",
        "## 3. 매출액과 유한 풀 기계적 기준",
        "",
        "`market_status.turnover_won`이 있으면 실제 경주×승식 매출액을 사용한다. 단, 매출액을 100원으로",
        "나눈 수를 독립 베팅 의사결정 수로 해석하지 않는다. 기계적 null에서만 100원 단위를 독립 다항",
        "추출로 가정한다. 이 null의 Monte Carlo 반복 수는 경주당",
        f"**{TURNOVER_MC_DRAWS}회**, 기준시드는 `{TURNOVER_MC_SEED}`이다.",
        "",
    ]
    if not turnover_summary.empty:
        lines += [
            "|승식|매출완비 clean 경주|관측 TV 중앙값|null TV 중앙값|관측/null 비율|관측이 경주별 null p95 초과 비율|",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, row in turnover_summary.sort_values("target_market").iterrows():
            lines.append(
                f"|{label[row['target_market']]}|{int(row['n_turnover_complete_clean_races']):,}|"
                f"{row['observed_median_tv']:.6f}|{row['median_mc_median_tv']:.6f}|"
                f"{row['observed_to_mc_median_ratio']:.2f}|{row['share_observed_gt_race_mc_p95']:.3f}|"
            )
        lines += [
            "",
            "이 값은 실제 잡음하한의 추정치가 아니라, 모든 100원 단위가 서로 독립이라는 강한 가정 아래의",
            "기계적 비교선이다. 따라서 관측 TV가 이 기준보다 크다는 사실은 유한 풀 잡음만으로 설명하기",
            "어렵다는 방향의 증거이지만, 실제 베터 수나 독립 베팅건수를 식별하지는 않는다.",
            "",
        ]
    if not limit_summary.empty and bool(limit_summary.iloc[0].get("available", False)):
        row = limit_summary.iloc[0]
        lines += [
            "한국 경마의 경주당 개인 구매상한 100,000원을 함께 고려하면, 모든 승식의 총매출을",
            "100,000원으로 나눈 올림값은 해당 경주에 필요한 최소 베터-경주 참여수의 하한이다.",
            f"표본 내 이 하한의 중앙값은 **{row['median_minimum_bettor_equivalents_at_100k_cap']:.0f}명 상당**이다.",
            "실제 이용자는 10만원을 여러 승식·조합에 나누어 베팅할 수 있으므로 100원 단위 독립성은",
            "현실보다 훨씬 강한 가정이며, 이 때문에 turnover-matched null은 보수적인 '낮은 잡음' 기준으로",
            "읽는 것이 적절하다.",
            "",
        ]
    lines += [
        "## 4. 재현 명령",
        "",
        "```bash",
        "python -m pip install -r requirements.txt -c constraints-behavioral.txt",
        "python -m unittest tests.test_cultural_industry_reverification -v",
        "python -m analysis.cultural_industry_reverification",
        "```",
        "",
        "## 5. 고정 산출물",
        "",
        "- `outputs/cultural_industry_reverification_summary.csv`: TV 중앙값과 두 bootstrap CI",
        "- `outputs/cultural_industry_reverification_per_race.csv`: 경주별 TV 재계산값",
        "- `outputs/cultural_industry_reverification_sample.csv`: 표본 수와 상한 수 재감사",
        "- `outputs/cultural_industry_2018_overcap_check.csv`: 2018-07-01의 9,999.9/초과배당 진단",
        "- `outputs/cultural_industry_turnover_limit_summary.csv`: 총매출과 10만원 상한의 하한 진단",
        "- `outputs/cultural_industry_turnover_null_summary.csv`: 실제 매출액 일치 기계적 null 요약",
        "- `outputs/cultural_industry_turnover_null_per_race.csv`: 경주별 null 결과",
        "- `outputs/cultural_industry_reverification_input_files.csv`: 사용 parquet 전부의 SHA-256",
        "- `outputs/cultural_industry_reverification_manifest.json`: 코드·환경·파라미터와 산출물 hash",
        "",
        f"입력 parquet 파일 수: **{len(input_hashes):,}개**.",
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(output_paths: list[Path], input_hashes: pd.DataFrame) -> None:
    combined = hashlib.sha256()
    for _, row in input_hashes.iterrows():
        combined.update(f"{row['path']}:{row['size_bytes']}:{row['sha256']}\n".encode("utf-8"))
    manifest = {
        "verification_date": "2026-08-16",
        "git_sha_at_run": os.getenv("GITHUB_SHA", "local"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "parameters": {
            "start_date": str(START_DATE.date()),
            "end_date": str(END_DATE.date()),
            "unavailable_years": sorted(UNAVAILABLE_YEARS),
            "display_cap": DISPLAY_CAP,
            "cap_definition": "odds == 9999.9 within 1e-9; values >9999.9 remain point observations",
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "turnover_mc_draws_per_race": TURNOVER_MC_DRAWS,
            "turnover_mc_seed": TURNOVER_MC_SEED,
            "nominal_wager_unit_won": NOMINAL_WAGER_UNIT_WON,
            "betting_limit_won_per_person_race": BETTING_LIMIT_WON_PER_PERSON_RACE,
        },
        "input_files": {
            "count": int(len(input_hashes)),
            "combined_sha256": combined.hexdigest(),
        },
        "outputs": {
            path.as_posix(): {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in output_paths
            if path.exists()
        },
    }
    path = OUTPUT_DIR / "cultural_industry_reverification_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    races = load_races()
    status = load_status()
    clean, sample_audit = build_clean_ids(races, status)
    common_ids = clean["win"]

    per_race = compute_per_race_tv(races, common_ids)
    tv_summary = summarize_tv(per_race)
    overcap = overcap_check(races)
    limit_summary = turnover_betting_limit_summary(status, races)
    turnover_race, turnover_summary = turnover_matched_null(status, common_ids)
    input_hashes = input_hash_inventory()

    sample_audit.to_csv(OUTPUT_DIR / "cultural_industry_reverification_sample.csv", index=False, float_format="%.12g")
    per_race.to_csv(OUTPUT_DIR / "cultural_industry_reverification_per_race.csv", index=False, float_format="%.12g")
    tv_summary.to_csv(OUTPUT_DIR / "cultural_industry_reverification_summary.csv", index=False, float_format="%.12g")
    overcap.to_csv(OUTPUT_DIR / "cultural_industry_2018_overcap_check.csv", index=False, float_format="%.12g")
    limit_summary.to_csv(OUTPUT_DIR / "cultural_industry_turnover_limit_summary.csv", index=False, float_format="%.12g")
    turnover_race.to_csv(OUTPUT_DIR / "cultural_industry_turnover_null_per_race.csv", index=False, float_format="%.12g")
    turnover_summary.to_csv(OUTPUT_DIR / "cultural_industry_turnover_null_summary.csv", index=False, float_format="%.12g")
    input_hashes.to_csv(OUTPUT_DIR / "cultural_industry_reverification_input_files.csv", index=False)

    write_doc(sample_audit, tv_summary, overcap, limit_summary, turnover_summary, input_hashes)
    output_paths = [
        OUTPUT_DIR / "cultural_industry_reverification_sample.csv",
        OUTPUT_DIR / "cultural_industry_reverification_per_race.csv",
        OUTPUT_DIR / "cultural_industry_reverification_summary.csv",
        OUTPUT_DIR / "cultural_industry_2018_overcap_check.csv",
        OUTPUT_DIR / "cultural_industry_turnover_limit_summary.csv",
        OUTPUT_DIR / "cultural_industry_turnover_null_per_race.csv",
        OUTPUT_DIR / "cultural_industry_turnover_null_summary.csv",
        OUTPUT_DIR / "cultural_industry_reverification_input_files.csv",
        DOC_PATH,
    ]
    write_manifest(output_paths, input_hashes)

    print(sample_audit.to_string(index=False))
    print(tv_summary.to_string(index=False))
    print(overcap.to_string(index=False))
    print(limit_summary.to_string(index=False))
    print(turnover_summary.to_string(index=False))


if __name__ == "__main__":
    main()
