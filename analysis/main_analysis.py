#!/usr/bin/env python3
"""Public entry point for the co-primary cross-pool main analysis."""
from analysis.main_analysis_core import (
    PriceSet,
    aggregate_price_set,
    harville_trifecta,
    normalize_inverse_odds,
    odds_to_price_set,
    point_metrics,
    point_price_set,
    price_set_component_bounds,
    source_group_index,
    tv_lower_exact,
    tv_upper_outer,
)
from analysis.main_analysis_runner import main

__all__ = [
    "PriceSet",
    "aggregate_price_set",
    "harville_trifecta",
    "normalize_inverse_odds",
    "odds_to_price_set",
    "point_metrics",
    "point_price_set",
    "price_set_component_bounds",
    "source_group_index",
    "tv_lower_exact",
    "tv_upper_outer",
    "main",
]


if __name__ == "__main__":
    main()
