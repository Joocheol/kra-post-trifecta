# Main-analysis outputs

`analysis/main_analysis.py` writes race-level Panel A point metrics and Panel B TV/MAE bounds here.
The full 19,284-race run completed successfully in PR #3. Large race-level CSVs remain generated
GitHub Actions artifacts. Compact Panel A/B summaries, benchmark-improvement summaries,
order-information tests, threshold decisions, clean-versus-capped selection/composition/tail
diagnostics, a compact heterogeneity summary, and manuscript tables are frozen in the repository.
Paper CI regenerates the full analysis once, regenerates the sample-selection manuscript table,
and rejects any change in the checked-in compact outputs.

The frozen full-run artifact used for the initial freeze was `main-analysis-full` from Paper CI run
31278482505 (artifact SHA-256 `5f89a5ea3f28b6a05086ba4ce2cec52413184f2676718d9580e41da5c9ce6aba`).
`outputs/main_analysis_manifest.json` is generated inside each CI artifact and records SHA-256 hashes
for the large race-level files and all files emitted by `analysis.main_analysis`; it is intentionally
not a separately frozen git file because a subsequent run also regenerates it. Repository-level
reproducibility is enforced instead by byte-for-byte checks on the compact frozen outputs listed in
`.github/workflows/paper-ci.yml`.
