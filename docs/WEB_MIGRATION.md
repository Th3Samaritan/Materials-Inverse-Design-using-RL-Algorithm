# Web migration verification record

Date: 30 September 2026.

## Delivered

The Streamlit presentation layer has been replaced by a React/Vite frontend and FastAPI service. Archived Streamlit sources are retained under `legacy/streamlit/` and excluded from the Docker image and active tests. Core physics, checkpoint formats and prediction logic remain in `mast/`.

The web UI retains solid white (`#FFFFFF`) surfaces and evergreen (`#24594F`) controls. It supports inverse searches, forward predictions, planned-family disclosure, equipment limits, candidate comparison, JSON export, Supabase email/password sign-in and paginated saved history. Editing inputs aborts the active fetch and invalidates stale results. Token refresh preserves an unchanged user's current work; changing users clears results and history. No remote font or image service is required.

The API enforces composition/target/domain validation and fixed 25 mm geometry, input and search bounds, serialized CPU inference, bearer-token verification, and explicit CORS origins. Production configuration rejects anonymous mode. The Supabase migration scopes records by `auth.uid()` and the API forwards the user's token for database access. Save errors do not discard successfully computed results.

Build/deployment artifacts: `Dockerfile`, `.dockerignore`, `requirements-api.txt`, `render.yaml`, `web/vercel.json`, frontend lockfile, example environments and additive Supabase SQL. The frontend hosted build validates its environment and rejects secret Supabase keys. There is no Streamlit server in this stack.

## Local checks performed

- **21 Python tests passed**: core/data tests plus HTTP tests with actual checkpoints, invalid requests, body limits, model-route protection, CORS, concurrency/rate limits, save failure, production sign-in and user-token forwarding.
- **6 frontend tests passed**: request payload/export, stale request suppression, unsupported family abstention, production sign-in gating, recoverable API errors and result display.
- **2 browser smoke tests passed in headless Edge** against the real local API: model search, JSON download, input invalidation, forward prediction, white page background, mobile viewport fit and unsupported-family abstention. Desktop, results and 390 px mobile screenshots were visually inspected. An initial screenshot write returned a transient disk-space error; retry succeeded with approximately 1.6 GB free reported by Windows.
- **Vite production build passed**, using explicitly fake HTTPS deployment values for compilation only. This is not a working cloud deployment and must not be uploaded as a release artifact.
- Native FastAPI and Vite development servers started successfully on local ports 8000 and 5173.

Python checks ran with PyTorch 2.12.1+cpu, FastAPI 0.138.1, Pydantic 2.13.2, Uvicorn 0.49.0 and httpx 0.28.1. Frontend dependencies are recorded in `web/package-lock.json`. The Dockerfile targets Python 3.13 and the same PyTorch version; Linux packaging has not been executed locally because Docker's daemon is unavailable.

## Still to verify with connected accounts

Actual Supabase migration execution and cross-user RLS isolation; real password sign-in; Render image build, startup memory and cold-start behavior; Vercel deployed origins and end-to-end saving. The app's browser connector could not initialize, so local browser inspection used Playwright/Edge instead; it did not provide access to the user's hosting-dashboard sessions.

This migration is application infrastructure work. It does not retrain or industrially validate the research model, repair old training data or add predictors for polymers/composites. Read the model card before interpreting any output.
