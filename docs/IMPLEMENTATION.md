# Implementation and verification record

Date: 8 September 2026.

Historical research/Streamlit implementation record. For the active React/FastAPI platform and current checks, see [Web migration](WEB_MIGRATION.md) and [Deployment](DEPLOYMENT.md). Streamlit source and its interaction tests have since been archived.

## Delivered changes

`mast/models.py` preserves the legacy checkpoint architectures outside Streamlit. `mast/catalog.py` centralizes the legacy feature/property order, synthetic support bounds and illustrative compositions.

`mast/engine.py` implements restricted checkpoint loading, normalization checks, model hashes, composition/target validation, equipment constraints, seeded policy/random/hybrid generation, post-generation screening, Pareto flags, deduplication and JSON-ready run reports. RNG sampling is serialized to protect reproducibility across concurrent app sessions. Existing weights and external training projects are unchanged.

Ranking minimizes mean squared target error standardized by the policy property scales, plus .002 times standardized ensemble variance and an optional normalized combined-hold-time penalty. This scalarization is an explicit preference, not a physics law. Pareto flags compare target mismatch, spread and hold time over the valid sampled set, before near-duplicate display filtering. They do not identify a global Pareto front.

The app excludes out-of-domain H13, exposes family readiness, provides constraints/search controls, labels research evidence, checks target tolerance and exports traceable runs. It clears stale recommendations on request/settings edits. The comparison plot uses percentage of each target so MPa, percentages, HRC and joules are not plotted as comparable raw magnitudes. Raw values remain in the detail table. Forward prediction uses the same screening.

`mast/physics.py` adds a bounded martensite-transformation primitive for retraining research. It does not repair existing weights or implement a full phase-kinetics model.

`mast/data.py` adds provenance/units checks and group-separated splits. `mast/train_baseline.py` evaluates measured-data random-forest and mean baselines in each property's original unit. It produces a report, not a deployed replacement model. `python -m mast` supplies a headless checkpoint demonstration. `scripts/benchmark_search.py` reproduces the local surrogate comparison.

## Verification

Run in the project environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q mast tests steel_heat_treatment_recommender.py
.\.venv\Scripts\python.exe scripts/benchmark_search.py
.\.venv\Scripts\python.exe -m mast --method hybrid --candidates 100 --output outputs/research_run.json
```

The suite covers actual-checkpoint inference, repeatable candidates without changing caller RNG state, hashes, no-feasible-process behavior, H13/composition rejection, invalid targets, equipment/geometry constraints, process/property screening, Pareto comparisons, martensite bounds, provenance/duplicate rejection, nonoverlapping groups, synthetic-data rejection and unit consistency. Streamlit interaction tests cover search, request invalidation, unsupported-family abstention and forward prediction.

The full suite passed **17 tests** on 8 September 2026: 14 core/data/checkpoint checks and 3 Streamlit interaction checks. Compilation and the headless JSON export also passed. These are software tests, not material-property validation. A browser layout check additionally exposed a legacy hidden-header rule that removed narrow-screen sidebar controls; the header controls were restored and initial sidebar behavior changed to automatic.

Verified environment: Python 3.13.13, PyTorch 2.12.1+cpu, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.7.1, Streamlit 1.63.0 and Plotly 6.8.0. This records tested versions, not a complete transitive dependency lock.

## Benchmark

`outputs/search_benchmark.json` records 27 runs: 3 presets × 3 seeds × 3 methods, each with 300 forward evaluations and the same default six-property target. CPU thread count is two; model loading is excluded. The file contains model hashes, timings, scores, valid-candidate counts and displayed tolerance matches.

Observed mean best scores: policy −.41624, random −.40049, hybrid −.40391. Approximate mean times: .055/.014/.037 seconds. All methods had zero displayed all-property tolerance matches in nine cases each. Random search was a stronger and faster comparator in this narrow test. Hybrid remains available to compare coverage from both generators; no default-method superiority claim is made.

This benchmark measures agreement with the same flawed legacy surrogate. It cannot demonstrate property accuracy, factory efficiency, lower experiment cost or true infeasibility. A prospective measured campaign is the decisive next evaluation.

## Remaining work

- Acquire permissioned, traceable measured datasets and agree on the first industrial pilot.
- Correct and independently validate the full steel physics generator; reconstruct training provenance.
- Add geometry/cooling-history/test-condition features and train calibrated family-specific models.
- Implement property intervals and minimum/maximum target semantics, total cycle/energy/cost models, proper OOD detection and calibrated joint uncertainty.
- Add active learning/BO only after benchmark data and uncertainty support it.
- Implement actual composite/polymer/elastomer adapters and models; the readiness catalog is not predictive support.
- Implement licensed simulation/DFT jobs, experiment ingestion/LIMS workflows and model promotion/rollback.
- Implement authentication, tenant isolation, persistent audit storage, billing and production operations before selling hosted access.

No industrial model was trained, no manufacturing recipe was qualified, no subscription service was deployed and no DFT calculation was performed. The delivered result is a tested research foundation plus an evidence-backed development plan, not a completed commercial platform.
