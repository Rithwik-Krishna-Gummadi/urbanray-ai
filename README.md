# UrbanRay AI

Curated current project — 15 November 2026.

**Current 88-building research project.** The latest physics source is **Physics+ v2 integrated into Validation v3**, plus the newer **2025–2026 Landsat date search**, **10-date simulation plan**, and **actual 10-scene exploratory results/audit**.

## Open first

- `UrbanRay_Latest_Colab.ipynb`: notebook for setup, verification, charts, and *optional* new simulations.
- `README.md`: this guide.
- `SCIENTIFIC_STATUS.md`: what the project demonstrates and what remains unverified.
- `outputs/multidate_validation/audit/AUDIT_REPORT.md`: independent arithmetic and cautions for ten observed satellite scenes.

## Quick start (local Python or Colab terminal)

```bash
python -m pip install -r requirements-tested.txt
python -m pytest -q
python UrbanRay_Run_10Date_Validation.py --all --dry-run
```

The project does **not** need more Earth Engine exports to display its currently archived validation results. To search for further overpasses, paste **only** `earth_engine/UrbanRay_Find_Matching_Landsat_Dates.js` into the **Earth Engine JavaScript Code Editor**, **not** a Python notebook.

## Current evidence

- Municipal NYC geometry: 88 documented buildings in `data/nyc_88_source.geojson`, with 2,082 simplified modeled surface patches.
- Archived real-weather-based multi-day physics simulation from June 2025 in `outputs/historical_ml/` (model input weather hashes are recorded, but **raw historical weather is NOT included**).
- Trained simulator surrogate and its held-out simulation-relative scores in `outputs/ml_holdout/`.
- Independently sourced Landsat exports: 150 candidate scenes, 44 shortlisted scenes, and 1,104 qualifying 30m output-grid rows, stored as the three `urbanray_*_2025_2026.csv` files at project root.
- Ten already-completed scene-level exploratory satellite-to-model comparisons in `outputs/multidate_validation/`, including five earlier 2025 development scenes and five 2026 held-out scenes.
- Held-out scene-level raw MAE **3.526 °C**; after additive correction learned from the five development scenes, **2.341 °C**. This is exploratory *proxy-versus-retrieval disagreement*, **not verified individual-roof temperature accuracy**. See audit limitations.

## Common workflows

**Explore existing 10-date results**

```python
import pandas as pd
print(pd.read_csv('outputs/multidate_validation/scene_level_EXPLORATORY.csv').to_string(index=False))
```

**Reproduce ray/geometry unit checks**

```bash
python -m pytest -q
```

**Inspect the selected 10 historical overpass dates, before running**

```bash
python UrbanRay_Run_10Date_Validation.py --all --dry-run
```

**Run one selected observation date with actual external historical weather**

```bash
python UrbanRay_Run_10Date_Validation.py --dates 2025-04-23
```

This re-downloads missing Open-Meteo gridded weather with source provenance, runs the Physics+ model with 48-hour warm-up and 16 rays, and compares the resulting face temperatures to the already-provided Landsat measurements. **Computationally expensive; internet is required.** The ten-date summarized result is already archived, so a rerun is *optional*, not required to see existing metrics.

**Regenerate training data and retrain the AI**

The large generated face table is excluded from GitHub. Download weather and run the fresh historical simulation below first, then train on its output. This is a new run, not a promise of bit-for-bit reproduction of the archived model.

```bash
python train_surrogate.py --faces outputs/new_historical_run/physics_training_faces.csv --out outputs/ml_retrained
```

**Launch a fresh historical simulation**

```bash
python fetch_weather_range.py --start 2025-06-18 --end 2025-06-25 --out data/nyc_weather_2025.csv
python run_physics_plus.py --weather-csv data/nyc_weather_2025.csv --warmup-hours 48 --hours 96 --rays 8 --output outputs/new_historical_run
```

## Contents

- `src/`, `run_physics_plus.py`, `fetch_weather_range.py`: current 88-building thermal/shortwave/longwave implementation.
- `train_surrogate.py`: AI trained to predict physics-simulator shortwave targets (not measured roof temperatures).
- `validate_landsat_pixels.py`, `evaluate_scene_holdout.py`, `UrbanRay_Run_10Date_Validation.py`: current satellite comparison and preselected dates.
- `earth_engine/`: one current date-search + filtered-pixel export (5 K exploratory threshold). Previous 3 K and incompatible-projection scripts intentionally excluded.
- `data/`: real geometry and archived ray test inputs.
- `tests/`: included unit tests; synthetic fixture CSVs are for tests only.
- `outputs/historical_ml/`: compact historical physics summaries and provenance; large generated face tables and redundant renders excluded.
- `outputs/ml_holdout/`: trained surrogate, held-out score summaries and audit; redundant row-level predictions excluded.
- `outputs/historical_sensitivity/`, `outputs/ray_convergence_multi_seed/`: sensitivity and numerical repeatability.
- `outputs/multidate_validation/`: observed-vs-modeled summary, heldout correction and audited figures.
- `CHECKSUMS_SHA256.json`: checksums for all package files except the checksum manifest itself.

## Completeness and limitations

This bundle includes **everything currently available here needed to read/reuse the latest source, observed Landsat exports, trained AI, and reported summary results**. It does **not** contain the raw Open-Meteo weather CSV/API responses for all ten satellite dates, the full per-date face-temperature outputs, or all pixel-level matches from the 10-date run; those were not included in the submitted multi-date results ZIP. Therefore, its ten-date simulation histories **cannot be independently rerun offline**. The supplied downloader/re-run script supports reproducing them when external weather is accessible. The 2026 test dates were subsequently inspected; do not claim further tuning on these scenes would be a pristine held-out test.

Data and geometry choices are model approximations. Landsat native thermal resolution (~100 m) is coarser than its 30 m product grid; rows in a single scene are correlated. No measured individual-building roofing material data; no ground-observed rooftop temperature series. Existing-roof observations cannot independently validate the hypothetical cool-roof intervention.

## Repository notes

See `CURATION.md` for exclusions and `VERIFICATION.md` for checks performed during preparation. `requirements-tested.txt` records the validation environment. The original ZIP is unchanged.

The Colab notebook accepts a GitHub Download ZIP or the curated distribution ZIP. Test weather and canyon fixtures are synthetic and are not observational evidence.

The saved surrogate requires **scikit-learn 1.6.1**, matching its training version; both requirements files enforce this. It was checked against all 20,976 archived held-out simulator rows.
