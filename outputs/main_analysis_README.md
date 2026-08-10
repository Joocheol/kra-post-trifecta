# Main-analysis outputs

`analysis/main_analysis.py` writes race-level Panel A point metrics and Panel B TV/MAE bounds here.
The full 19,284-race run completed successfully in PR #3. Large race-level CSVs and long-form
heterogeneity decompositions remain generated GitHub Actions artifacts. Compact Panel A/B summaries,
benchmark-improvement summaries, order-information tests, threshold decisions, sample-selection and
composition summaries, the `other_race` donor-reuse diagnostic, and Panel A/B heterogeneity comparison
summaries/tables are frozen in the repository. Paper CI regenerates the full analysis once and rejects
changes in the manuscript-facing checked-in compact outputs, providing a deterministic result-freeze
check without a second full run.

`main_other_race_donor_reuse.csv` is an auxiliary benchmark-dependence diagnostic generated with the
full analysis and frozen in the repository. It reports, by panel and field-size stratum, the number of
target races with a donor, the number of distinct deterministic donors, targets per distinct donor,
and maximum donor reuse. Because several target races can share the same donor price vector, the
ordinary race bootstrap for the `other_race` benchmark can understate dependence in thin field-size
strata. This diagnostic does not replace the pre-specified race bootstrap, alter the co-primary
decision rule, or introduce a new primary inference procedure; it only makes the benchmark dependence
visible.

The benchmark-improvement CSVs retain one shared legacy column name,
`median_improvement_lower`. In Panel A this field is the median of the exact point-improvement
quantity because Panel A uses point prices. In Panel B it is the median of the conservative lower
improvement quantity constructed from the partial-identification bounds. The common name is retained
to preserve the frozen output schema; the LaTeX table labels the displayed quantity generically as
`개선폭 중앙값` and the panel indicator determines which definition applies.

The historical `main-analysis-full` artifact from Paper CI run 31278482505 had SHA-256
`5f89a5ea3f28b6a05086ba4ce2cec52413184f2676718d9580e41da5c9ce6aba`. This artifact reference is
provenance only: the manifest and large race-level files are CI artifacts rather than independent
Git-tracked verification objects. Repository-level reproducibility is enforced by regenerating and
checking the manuscript-facing compact outputs in Paper CI.
