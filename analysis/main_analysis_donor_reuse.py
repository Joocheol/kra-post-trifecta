"""Auxiliary dependence diagnostic for the deterministic other-race benchmark.

This module does not change the co-primary inference.  It only summarizes how
many target races share the same deterministic same-field donor, because the
ordinary race bootstrap treats race-level benchmark improvements as if they
were independently resampled even when donor price vectors are reused.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def donor_reuse_diagnostic(
    panel_a_metrics: pd.DataFrame,
    panel_b_bounds: pd.DataFrame | None,
) -> pd.DataFrame:
    """Summarize deterministic donor reuse by panel and field-size stratum."""
    frames: list[tuple[str, pd.DataFrame]] = [("A", panel_a_metrics)]
    if panel_b_bounds is not None:
        frames.append(("B", panel_b_bounds))

    rows: list[dict[str, object]] = []
    for panel, frame in frames:
        required = {"race_id", "model", "n_valid_horses", "donor_race_id"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{panel}: donor diagnostic missing columns {sorted(missing)}")
        donors = frame[frame["model"].eq("other_race")][
            ["race_id", "n_valid_horses", "donor_race_id"]
        ].copy()
        donors["donor_race_id"] = donors["donor_race_id"].fillna("").astype(str)
        donors = donors[donors["donor_race_id"].str.len() > 0]
        donors = donors.drop_duplicates()
        per_race = donors.drop_duplicates(subset=["race_id"], keep="first")
        if len(per_race) != donors["race_id"].nunique():
            raise ValueError(f"{panel}: one target race maps to multiple donors")

        for field_size, group in per_race.groupby("n_valid_horses", sort=True):
            counts = group["donor_race_id"].value_counts()
            n_targets = int(len(group))
            n_donors = int(counts.size)
            rows.append(
                {
                    "panel": panel,
                    "n_valid_horses": int(field_size),
                    "n_targets_with_donor": n_targets,
                    "n_distinct_donors": n_donors,
                    "targets_per_distinct_donor": (
                        float(n_targets / n_donors) if n_donors else np.nan
                    ),
                    "max_targets_per_donor": int(counts.max()) if n_donors else 0,
                }
            )
    return pd.DataFrame(rows).sort_values(["panel", "n_valid_horses"]).reset_index(drop=True)
