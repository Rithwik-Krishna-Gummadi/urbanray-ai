# Verification — 15 November 2026

- All 62 source-file SHA-256 checksums matched the supplied archive manifest before curation.
- Final environment: Python 3.12 on Windows, recorded in requirements-tested.txt.
- 29/29 automated tests passed after removing the legacy simulator and pinning scikit-learn to 1.6.1. Shapely deprecation warnings remain; they did not fail tests.
- Saved HistGradientBoostingRegressor loaded successfully with scikit-learn 1.6.1. Predictions on all 20,976 original held-out rows reproduced the archived simulator-relative MAE: 4.1781429500803755 W/m². The original large table was used locally for verification and is excluded from the repository.
- One-hour end-to-end Physics+ smoke run completed for 88 real building geometries and two roof scenarios using clearly labeled synthetic test weather; this is a software check, not real-world validation.
- Ten-date dry run recognized all ten planned dates.
- Scene evaluation recomputed raw held-out MAE 3.5261483184561717 °C and corrected MAE 2.3413728337847823 °C from archived summaries.
- Python syntax checks and a credential-pattern scan completed; no credential candidates were found in the curated files.

Full ten-date weather acquisition and physics reruns were not performed. Missing original weather/per-date face data prevents exact offline reproduction. The satellite comparison remains exploratory; see SCIENTIFIC_STATUS.md.
