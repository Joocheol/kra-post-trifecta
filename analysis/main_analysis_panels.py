"""Data loading and race-level Panel A/B construction for the main analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from analysis.data_audit import MARKET_SPECS, read_parquets
from analysis.main_analysis_core import (
    EPSILON,
    PriceSet,
    aggregate_point,
    aggregate_price_set,
    choose_other_race,
    deterministic_permutation,
    harville_trifecta,
    normalize_inverse_odds,
    odds_to_price_set,
    point_metrics,
    point_price_set,
    source_group_index,
    target_key,
    target_keys_from_frame,
    tv_upper_outer,
)
from analysis.main_analysis_fast import tv_lower_exact_fast


@dataclass(frozen=True)
class RaceSlices:
    frame: pd.DataFrame
    slices: Mapping[str, tuple[int, int]]

    def get(self, race_id: str) -> pd.DataFrame:
        start, count = self.slices[race_id]
        return self.frame.iloc[start : start + count]


def market_sort_columns(market: str) -> list[str]:
    return ["race_id", *MARKET_SPECS[market].keys]


def load_market(data_root, market: str, race_ids: set[str]) -> RaceSlices:
    spec = MARKET_SPECS[market]
    columns = ["race_id", *spec.keys, "odds", "is_capped_odds"]
    frame = read_parquets(data_root, market, columns=columns)
    frame = frame[frame["race_id"].isin(race_ids)].copy()
    frame = frame.sort_values(market_sort_columns(market)).reset_index(drop=True)
    ids = frame["race_id"].to_numpy()
    unique, starts, counts = np.unique(ids, return_index=True, return_counts=True)
    slices = {
        str(race_id): (int(start), int(count))
        for race_id, start, count in zip(unique, starts, counts)
    }
    missing = race_ids.difference(slices)
    if missing:
        raise ValueError(f"{market}: missing {len(missing)} eligible races")
    return RaceSlices(frame=frame, slices=slices)


def grouped_ids_by_field(races: pd.DataFrame, race_ids: Iterable[str]) -> dict[int, list[str]]:
    subset = races[races["race_id"].isin(set(race_ids))][["race_id", "n_valid_horses"]]
    return {
        int(n): sorted(group["race_id"].astype(str).tolist())
        for n, group in subset.groupby("n_valid_horses")
    }


def interval_for_frame(frame: pd.DataFrame) -> PriceSet:
    return odds_to_price_set(
        frame["odds"].to_numpy(dtype=float),
        frame["is_capped_odds"].fillna(False).to_numpy(dtype=bool),
    )


def panel_a(
    races: pd.DataFrame,
    trifecta: RaceSlices,
    win: RaceSlices,
    target_name: str,
    target: RaceSlices,
    clean_ids: list[str],
    clean_peers: dict[int, list[str]],
) -> pd.DataFrame:
    """Compute Panel A point metrics; donor benchmark may be unavailable by stratum."""
    rows: list[dict[str, object]] = []
    n_map = races.set_index("race_id")["n_valid_horses"].astype(int).to_dict()
    race_lookup = races.set_index("race_id")
    for race_id in clean_ids:
        source = trifecta.get(race_id)
        actual_frame = target.get(race_id)
        win_frame = win.get(race_id)
        groups = source_group_index(source, actual_frame, target_name)
        cdim = len(actual_frame)
        q_main = normalize_inverse_odds(source["odds"].to_numpy(dtype=float))
        actual = normalize_inverse_odds(actual_frame["odds"].to_numpy(dtype=float))
        q_h = harville_trifecta(source, win_frame)
        q_uniform = np.full(len(source), 1.0 / len(source), dtype=float)
        permutation = deterministic_permutation(len(source), race_id, target_name, "A")
        q_perm = q_main[permutation]
        n = int(n_map[race_id])
        arrival = tuple(int(value) for value in race_lookup.at[race_id, "arrival_tuple"][:3])
        realized_key = target_key(arrival, target_name)
        target_keys = target_keys_from_frame(actual_frame, target_name)
        try:
            realized_index = target_keys.index(realized_key)
        except ValueError as exc:
            raise ValueError(
                f"realized {target_name} outcome is absent for race {race_id}"
            ) from exc
        race_date = str(race_lookup.at[race_id, "race_date"])

        predictions: dict[str, tuple[np.ndarray, str]] = {
            "main": (aggregate_point(q_main, groups, cdim), ""),
            "harville": (aggregate_point(q_h, groups, cdim), ""),
            "permutation": (aggregate_point(q_perm, groups, cdim), ""),
            "uniform": (aggregate_point(q_uniform, groups, cdim), ""),
        }
        donor_id = choose_other_race(race_id, clean_peers.get(n, []), "A")
        if donor_id is not None:
            donor_source = trifecta.get(donor_id)
            if len(donor_source) != len(source):
                raise ValueError("same-field donor has different trifecta support size")
            q_donor = normalize_inverse_odds(donor_source["odds"].to_numpy(dtype=float))
            predictions["other_race"] = (
                aggregate_point(q_donor, groups, cdim),
                donor_id,
            )

        for model, (predicted, used_donor) in predictions.items():
            rec: dict[str, object] = {
                "panel": "A",
                "race_id": race_id,
                "race_date": race_date,
                "target_market": target_name,
                "model": model,
                "n_valid_horses": n,
                "n_outcomes": cdim,
                "donor_race_id": used_donor,
            }
            rec.update(point_metrics(actual, predicted))
            rec["realized_log_score"] = -float(
                np.log(max(float(predicted[realized_index]), EPSILON))
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def panel_b(
    races: pd.DataFrame,
    trifecta: RaceSlices,
    win: RaceSlices,
    target_name: str,
    target: RaceSlices,
    full_ids: list[str],
    full_peers: dict[int, list[str]],
) -> pd.DataFrame:
    """Compute Panel B exact TV lower and certified outer upper bounds."""
    rows: list[dict[str, object]] = []
    n_map = races.set_index("race_id")["n_valid_horses"].astype(int).to_dict()
    for race_id in full_ids:
        source = trifecta.get(race_id)
        actual_frame = target.get(race_id)
        win_frame = win.get(race_id)
        groups = source_group_index(source, actual_frame, target_name)
        cdim = len(actual_frame)
        actual_set = interval_for_frame(actual_frame)
        source_set = interval_for_frame(source)
        main_set = aggregate_price_set(source_set, groups, cdim)

        q_h = harville_trifecta(source, win_frame)
        harville_set = point_price_set(aggregate_point(q_h, groups, cdim))
        uniform_set = point_price_set(np.full(cdim, 1.0 / cdim, dtype=float))
        permutation = deterministic_permutation(len(source), race_id, target_name, "B")
        perm_source = PriceSet(
            source_set.lower[permutation], source_set.upper[permutation]
        )
        permutation_set = aggregate_price_set(perm_source, groups, cdim)
        n = int(n_map[race_id])

        models: dict[str, tuple[PriceSet, str]] = {
            "main": (main_set, ""),
            "harville": (harville_set, ""),
            "permutation": (permutation_set, ""),
            "uniform": (uniform_set, ""),
        }
        donor_id = choose_other_race(race_id, full_peers.get(n, []), "B")
        if donor_id is not None:
            donor_source = trifecta.get(donor_id)
            if len(donor_source) != len(source):
                raise ValueError("same-field donor has different trifecta support size")
            donor_set = aggregate_price_set(interval_for_frame(donor_source), groups, cdim)
            models["other_race"] = (donor_set, donor_id)

        for model, (prediction_set, used_donor) in models.items():
            lower = tv_lower_exact_fast(actual_set, prediction_set)
            upper = tv_upper_outer(actual_set, prediction_set)
            if lower - upper > 1e-7:
                raise RuntimeError(
                    f"TV bounds inverted for {race_id} {target_name} {model}"
                )
            rows.append(
                {
                    "panel": "B",
                    "race_id": race_id,
                    "target_market": target_name,
                    "model": model,
                    "n_valid_horses": n,
                    "n_outcomes": cdim,
                    "donor_race_id": used_donor,
                    "tv_lower": lower,
                    "tv_upper_outer": upper,
                    "mae_lower": 2.0 * lower / cdim,
                    "mae_upper_outer": 2.0 * upper / cdim,
                }
            )
    return pd.DataFrame(rows)
