# Repository curation — 15 November 2026

Source: UrbanRay_Latest_Complete_2026-11-15.zip. All 62 original manifest checksums verified before editing.

Removed the superseded synthetic 420-building simulator; retained its three required thermal correlation functions unchanged in src/thermal_correlations.py. The small synthetic canyon and weather fixtures remain only for software tests.

Excluded the 131 MB generated training table, row-level ML predictions, redundant HTML/geometry renders, and the old ZIP-specific audit utility. Trained model, score summaries, provenance audits, geometry, satellite CSVs, and current validation results remain. Original ZIP is unchanged.

The omitted face table can be regenerated with the weather-download and physics commands in README.md. Re-downloaded weather may differ from the archived run; exact historical model retraining is not promised. Raw inputs for the ten-date historical runs were not in the source ZIP.

No license was supplied in the source package; this upload does not add or infer one.
