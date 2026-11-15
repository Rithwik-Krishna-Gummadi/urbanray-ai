# UrbanRay 10-date observational comparison — independent file audit

**Input:** `UrbanRay_MultiDate_Validation.zip` (4 ZIP entries: the main scene CSV, held-out scene CSV, held-out JSON, and a directory).

## Verified numerical results

- Total scenes / acquisition dates: **10**, split into **5 development scenes in 2025** and **5 held-out scenes in 2026**.
- Matched 30-meter output-grid records across all scenes: **272**. These are correlated grid records, not independent native-resolution Landsat thermal observations.
- The 2025 development residual mean was **+1.431 °C**, so the reported additive calibration of **-1.431 °C** is arithmetically correct.
- 2026 held-out **raw**: MAE **3.526 °C**, RMSE **4.061 °C**, bias **+3.526 °C**.
- 2026 held-out **calibrated**: MAE **2.341 °C**, RMSE **2.907 °C**, bias **+2.095 °C**.
- Raw MAE improvement after calibration: **33.6%**.
- The mean per-scene Landsat ST_QA uncertainty across held-out scenes is **3.034 K**. Do not treat this as uncertainty of the scene mean or a justified error tolerance.

## Held-out 2026 scene residuals (model minus observation)

| Date | 30 m grid cells | Raw residual °C | Corrected residual °C |
|---|---:|---:|---:|
| 2026-05-12 | 27 | +1.696 | +0.264 |
| 2026-06-05 | 25 | +6.167 | +4.735 |
| 2026-06-13 | 29 | +5.120 | +3.689 |
| 2026-07-08 | 27 | +0.815 | -0.616 |
| 2026-07-24 | 29 | +3.834 | +2.402 |

## Interpretation and limitations

1. All five held-out raw residuals are positive; the model's simplified emission proxy runs warm relative to Landsat on every test scene. The largest disagreements were June 5 and June 13, 2026.
2. Bias correction helps but a residual positive bias remains. With only five test scenes, performance across seasons and other neighborhoods cannot be inferred.
3. The scenes were *labeled* development/held-out in the ZIP; no prior signed preregistration record was provided. We checked the partition and arithmetic, not that the split was chosen before analysis.
4. Model emission proxies are not equivalent to Landsat thermal retrievals. Satellites have a substantially coarser native thermal resolution than the 30 m output grid. Mixed surfaces, atmospheric correction, emissivity, wall visibility and vegetation add structural errors.
5. The ZIP includes **only aggregated CSVs and JSON**—no raw per-pixel matches, original satellite exports, model face-level hourly temperatures, historical weather files, execution logs, or complete configs. We cannot independently reproduce the simulations or verify scene matching and QA from this archive alone.
6. Temperature discrepancies do not validate the hypothetical reflective-roof cooling effect. The observations are of existing land surfaces.

## Next checks

- Preserve this 2026 set as untouched evaluation data; make no further adjustments using these held-out residuals before evaluating new independent scenes.
- Inspect June 5/June 13 weather input, cloud screening, thermal warmup, albedo and emissivity assumptions. Re-run independent scenes after any physics change; do not report new scores as the original held-out evaluation.
- Archive full input provenance and source code, matched pixel table, model temperatures at satellite UTC acquisition, sensor quality masks and scene overlap diagnostics. Add independent dates/areas and a simple physics baseline.
