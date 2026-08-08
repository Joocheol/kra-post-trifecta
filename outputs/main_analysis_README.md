# Main-analysis outputs

`analysis/main_analysis.py` writes race-level Panel A point metrics and Panel B TV/MAE bounds here.
The full 19,284-race run completed successfully in PR #3. Large race-level CSVs remain generated
GitHub Actions artifacts. Compact Panel A/B summaries, benchmark-improvement summaries,
order-information tests, threshold decisions, sample-selection summary, and manuscript tables are
frozen in the repository. Paper CI regenerates the full analysis once and rejects any change in
these checked-in compact outputs, providing a deterministic result-freeze check without a second
full run.

The frozen full-run artifact used for the initial freeze was `main-analysis-full` from Paper CI run
31278482505 (artifact SHA-256 `5f89a5ea3f28b6a05086ba4ce2cec52413184f2676718d9580e41da5c9ce6aba`).
