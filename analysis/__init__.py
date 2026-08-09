"""Reproducible empirical analysis for the KRA post-trifecta paper.

The frozen design treats the Harville benchmark as conditional on uncapped
published win odds.  Install that invariant at the shared numerical primitive
so every caller, including the standalone P3 joint diagnostics, fails loudly
instead of silently point-valuing a capped win observation.
"""
from __future__ import annotations

from functools import wraps


def _install_harville_win_guard() -> None:
    from analysis import main_analysis_core as core
    from analysis.main_analysis_guards import assert_win_uncapped

    original = core.harville_trifecta
    if getattr(original, "_win_uncapped_guarded", False):
        return

    @wraps(original)
    def guarded_harville_trifecta(source, win):
        assert_win_uncapped(win)
        return original(source, win)

    guarded_harville_trifecta._win_uncapped_guarded = True
    core.harville_trifecta = guarded_harville_trifecta


_install_harville_win_guard()
