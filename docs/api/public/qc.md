# Quality Control

Technical quality control (QC) for extracted tiles: `is_outlier` marks clear
technical failures (corruption, blown-out exposure, near-constant images),
never biological or statistical rarity. Genuinely different, but valid,
tissue (tumor, stroma, adipose, necrosis, mucin, lymphocyte-rich regions,
...) is never treated as a failure by itself. `needs_review` tiles are kept
by default - `quality_control` (and the `clean` CLI command) never deletes
or moves tile files.

::: histoslice.qc.QCConfig

::: histoslice.qc.quality_control
