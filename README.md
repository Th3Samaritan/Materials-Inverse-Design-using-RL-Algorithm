# MAST Materials Inverse Design

A research platform for specifying material-property targets and screening processing candidates. The current predictor covers a **synthetic low-alloy steel quench-and-temper domain**, not an industrially qualified process-design system.

The September 2026 audit found a phase-fraction defect in the legacy training sources. Existing checkpoints are retained for reproducible research, clearly labeled, and must not be used as evidence of manufacturing performance. More training on the same flawed generator will not resolve this.

## Web platform

The active application is now **React/Vite on Vercel**, **FastAPI/PyTorch on Render**, and **Supabase Auth + Postgres** for accounts and private run history. Streamlit is archived under `legacy/streamlit/` and is not a runtime dependency. The interface keeps a solid white background and evergreen accents.

Deployment files are prepared; cloud resources have **not yet been deployed or connected**. See [Deployment](docs/DEPLOYMENT.md) for exact environment variables, setup order and verification.

## Run locally

Backend, from the repository root (Python 3.13):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:APP_ENV='development'
$env:ALLOW_ANONYMOUS='true'
$env:CORS_ORIGINS='http://127.0.0.1:5173,http://localhost:5173'
.\.venv\Scripts\python.exe -m uvicorn api.main:create_app --factory --host 127.0.0.1 --port 8000
```

Frontend, in a second terminal (Node 24):

```powershell
cd web
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. Local anonymous mode permits inference and JSON export; saving requires a configured Supabase account. `.env.example` files document configuration, but backend variables must be exported in the shell (the backend does not automatically load `.env`).

The API loads `checkpoints/forward_model.pt` and `checkpoints/policy.pt`. Restricted loading rejects arbitrary Python checkpoint objects. The web app does not accept checkpoint uploads; model changes go through a backend release.

Core commands and tests:

```powershell
.\.venv\Scripts\python.exe -m mast --method hybrid --candidates 100 --output outputs/research_run.json
.\.venv\Scripts\python.exe scripts/benchmark_search.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd web
npm test
```

With both local servers running, run `npm run test:e2e` from `web/` for real-browser checks. Install Chromium with `npx playwright install chromium`, or on Windows use the installed Edge browser by setting `$env:PLAYWRIGHT_CHANNEL='msedge'` first. Screenshots and test output are kept in the ignored `.cache/` directory.

## Implemented

- Three illustrative low-alloy presets and custom composition with explicit training-domain checks. H13 is excluded instead of silently clipped.
- Policy, random and hybrid candidate generation with reproducible seeds.
- Equipment temperature, allowed quench media and combined hold-time constraints.
- Finite/physical output screening, target-tolerance status, Pareto tradeoff flags and optional shorter-hold preference.
- JSON export with inputs, settings, screening counts, warnings and checkpoint hashes.
- Forward prediction with the same domain/equipment screening.
- Material-family readiness catalog. Tool/stainless steels, composites, thermoplastics and elastomers have documented extension requirements; their predictors are not implemented.
- Provenance-aware measurement records, group-separated splits and a measured-data baseline evaluation command.
- A bounded martensite primitive for future physics work. It does not change existing checkpoint predictions.

A score is a research ranking, not a success probability. Ensemble spread is not a calibrated prediction interval. Hold time omits ramps, transfers, quench and cooling, so it is not total cycle time or energy. A failed search does not prove physical impossibility.

## Documentation

- [Audit and model card](docs/AUDIT_AND_MODEL_CARD.md): architecture, data lineage, defects and limits.
- [Research and industrial roadmap](docs/RESEARCH_AND_ROADMAP.md): family physics, DFT strategy, customer problems and commercialization gates.
- [Data and validation](docs/DATA_AND_VALIDATION.md): measurement contract, evaluation and retraining process.
- [Implementation and verification](docs/IMPLEMENTATION.md): changes, tests, benchmark and remaining work.
- [Research sources](docs/SOURCES.md): papers, technical documentation and labeled vendor sources.
- [Deployment](docs/DEPLOYMENT.md): Supabase, Render and Vercel configuration, access controls, verification and rollback.
- [Web migration record](docs/WEB_MIGRATION.md): implementation details and tested versus unverified deployment behavior.

The external `MAST Consolidate/MAST-RL` project was inspected read-only. Its notebooks, experiments and model files were not altered.
