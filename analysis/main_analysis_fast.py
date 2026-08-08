"""Exact performance shortcuts for Panel B set-distance calculations."""
from __future__ import annotations

import numpy as np

from analysis.main_analysis_core import PriceSet, tv_lower_exact as tv_lower_exact_lp


def price_sets_intersect(left: PriceSet, right: PriceSet) -> bool:
    """Return whether two normalized interval boxes intersect.

    Write p=s*x=t*y with x in [l,u], y in [a,b] and r=t/s. Coordinatewise
    feasibility is l_i <= r b_i and r a_i <= u_i. Hence a common normalized
    vector exists iff max_i l_i/b_i <= min_i u_i/a_i, treating a_i=0 as no
    upper restriction on r. This is exact and avoids solving an LP when the
    minimum TV distance is known to be zero.
    """
    if left.size != right.size:
        raise ValueError("price sets must have the same dimension")
    lower_ratio = np.max(left.lower / right.upper)
    positive = right.lower > 0
    if np.any(positive):
        upper_ratio = np.min(left.upper[positive] / right.lower[positive])
    else:
        upper_ratio = np.inf
    return bool(lower_ratio <= upper_ratio + 1e-12)


def tv_lower_exact_fast(left: PriceSet, right: PriceSet) -> float:
    if price_sets_intersect(left, right):
        return 0.0
    return tv_lower_exact_lp(left, right)
