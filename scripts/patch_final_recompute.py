#!/usr/bin/env python3
"""Apply the two durable fixes discovered during the 19,301-race dry run."""

from pathlib import Path


def main() -> None:
    path = Path("analysis/cultural_industry_final_recompute.py")
    text = path.read_text(encoding="utf-8").replace("race_date_meta", "race_date")

    old = '''    for field_size, group in hit.groupby("n_valid_horses", sort=True):
        share, lo, hi, _ = mean_with_ci(
            full_plan, group["winning_trifecta_capped"], group["race_date"]
        )
        by_field_rows.append(
'''
    new = '''    for field_size, group in hit.groupby("n_valid_horses", sort=True):
        # Subgroup inference resamples only race dates represented in the subgroup;
        # using the full 1,130-date universe can generate empty subgroup draws.
        subgroup_plan = ClusterBootstrapPlan.build(
            group["race_date"].astype(str).unique()
        )
        share, lo, hi, _ = mean_with_ci(
            subgroup_plan, group["winning_trifecta_capped"], group["race_date"]
        )
        by_field_rows.append(
'''
    if old in text:
        text = text.replace(old, new, 1)
        marker = '''                "ci_high": hi,
            }
        )
    return overall, pd.DataFrame(by_year_rows), pd.DataFrame(by_field_rows)
'''
        replacement = '''                "ci_high": hi,
            }
        )
        del subgroup_plan
    return overall, pd.DataFrame(by_year_rows), pd.DataFrame(by_field_rows)
'''
        if marker not in text:
            raise SystemExit("field-size block ending not found")
        text = text.replace(marker, replacement, 1)
    elif "subgroup_plan = ClusterBootstrapPlan.build(" not in text:
        raise SystemExit("field-size subgroup patch target not found")

    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
