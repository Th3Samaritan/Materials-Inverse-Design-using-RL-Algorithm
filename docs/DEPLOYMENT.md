# MAST deployment: Supabase, Render and Vercel

Prepared 30 September 2026. **Local implementation verified; cloud deployment pending.**

Deployment source: branch `codex/supabase-render-vercel` in this repository. Select this branch in Render/Vercel until the migration is merged into `main`; the previous `main` application is Streamlit.

## Architecture

| Component | Host | Responsibility |
|---|---|---|
| `web/` React/Vite | Vercel | White-and-green UI, sign-in, request controls, charts, JSON export |
| `api/` FastAPI | Render | Validate user tokens, bounded CPU inference, model loading, authenticated persistence |
| `supabase/migrations/` | Existing Supabase project | Auth users and the new `public.mast_runs` table with row-level security |

Use the same hosting accounts as Mineral-ML, with **separate MAST Render and Vercel services**. The migration only adds `mast_runs`; it does not modify Mineral-ML tables, storage or services. Reusing the Supabase project also reuses its user directory: any authenticated project user may use MAST. Per-product entitlement, organization roles and subscriptions are not implemented.

Model weights stay in the API image and are not served by its routes or bundled with the frontend. They are already tracked in this public GitHub repository; this split does not make those existing weights proprietary.

## 1. Supabase

1. Select the existing project used for Mineral-ML. Record its project URL and **publishable key** (a legacy `anon` key also works). Do not use a secret or `service_role` key.
2. Review and execute `supabase/migrations/202609300001_mast_runs.sql` once in that project's SQL editor, or apply it with your established migration workflow. Do not reset the shared database. Normal versioned migration execution is expected; repeated manual execution will report that objects already exist.
3. Use existing confirmed users or create a confirmed email/password test user through the administrator dashboard. The current UI supports email/password sign-in; it does not implement public signup, email-link callbacks or password recovery. Keep the existing Mineral-ML Auth Site URL and redirects unchanged.
4. Verify the cross-user tests below before public release. RLS is defined in the migration but has not been exercised against your live project yet.

The API calls `/auth/v1/user` for each protected request to verify the bearer token. Database requests forward that same user token; ownership policies remain active. The API does not use an administrative bypass key. Supabase recommends explicit RLS policies for browser-accessible tables: [Row Level Security](https://supabase.com/docs/guides/database/postgres/row-level-security).

## 2. Render

Create a new Blueprint using this repository and `render.yaml`. Service name: `mast-materials-api`. Docker context: repository root. The Dockerfile uses a CPU PyTorch installation, a non-root user, one Uvicorn worker, and Render's `PORT` variable. Health path: `/health`. Follow [Render's FastAPI guide](https://render.com/docs/deploy-fastapi) and [Blueprint specification](https://render.com/docs/blueprint-spec).

Set these service variables:

| Variable | Value |
|---|---|
| `APP_ENV` | `production` (Blueprint default) |
| `ALLOW_ANONYMOUS` | `false` (Blueprint default; never enable on Render) |
| `TORCH_NUM_THREADS` | `2` |
| `SUPABASE_URL` | Existing project's HTTPS URL |
| `SUPABASE_PUBLISHABLE_KEY` | Same project's publishable or legacy anon key |
| `CORS_ORIGINS` | Exact Vercel production origin, e.g. `https://your-mast-project.vercel.app`; no trailing slash |

If the Vercel domain is not assigned yet, create the Vercel project first, then return here. You can initially use `http://127.0.0.1:5173` as a restricted development origin and replace it after Vercel assigns the production URL. Do not use `*`. Comma-separated additional origins are supported; add specific preview origins only when needed.

The Blueprint chooses the free instance for an initial preview, not an industrial production commitment. Render documents idle spin-down, cold starts, ephemeral storage and shared free-hour limits. Two services in the same workspace share those limits. Check capacity and choose an appropriate paid instance before offering uptime commitments: [Render free service limits](https://render.com/docs/free). No paid plan was provisioned during this migration.

`GET /health` must report `model_ready: true`. If boot fails, inspect build logs, CPU-wheel availability and service memory before changing model loading. Do not enable unsafe checkpoint deserialization.

## 3. Vercel

Import the same GitHub repository as a **new** MAST project. Set Root Directory to `web`, Framework Preset to Vite, Install Command to `npm ci`, Build Command to `npm run build`, Output Directory to `dist`, and Node.js to 24. Do not deploy the repository root or replace Mineral-ML's existing project. See [Vite on Vercel](https://vercel.com/docs/frameworks/frontend/vite).

Configure these build-time environment variables in each deployment environment you intend to use:

| Variable | Value |
|---|---|
| `VITE_API_URL` | Actual Render API HTTPS origin |
| `VITE_SUPABASE_URL` | Same Supabase HTTPS project URL used by the API |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Same public publishable/anon key |

These values are intentionally visible to browser users. Never put private keys into `VITE_` variables. Hosted builds fail if values are absent, URLs are not HTTPS, or the supplied key looks like a Supabase secret/service-role key. Changing variables requires a new frontend build.

After Vercel assigns the domain, update Render's `CORS_ORIGINS`. For CLI deployment after account authentication, run `vercel link` and `vercel --prod` **from `web/`**, explicitly selecting the new MAST project. No CLI deployment has been performed yet. Never publish the local test build: it used placeholder hostnames solely to exercise compilation.

## Acceptance checks on the real deployment

1. Check `/health` and `/api/catalog` on Render. The catalog must contain `auth_required: true`.
2. A POST to `/api/recommend` without a bearer token must return 401; so must an expired or forged token. `/checkpoints/forward_model.pt` must return 404.
3. Sign in on Vercel as user A, run AISI 4340 defaults, export JSON, then select save and run again. Verify the saved record survives a Render restart. Inputs, model hashes and research warnings must be present.
4. Sign out and sign in as user B. User A's record must not appear. Direct PostgREST reads with B's token must not return it, and an attempted insert with A's `user_id` must fail. An anonymous request must not read or insert records. These checks test actual database policies, not just API behavior.
5. Change targets while a request is in flight. The old results must not reappear. Check narrow-screen layout and keyboard focus; verify unsupported material families offer no prediction button.
6. Temporarily test save failure in a staging environment: inference results should remain exportable with a visible save error. Do not disrupt shared production Supabase for this test.
7. Measure cold-start time, warmed search latency and process memory on the selected Render instance. Local software tests do not establish production load capacity.

## Operations and limits

- API contracts: `GET /health`, `GET /api/catalog`, authenticated `POST /api/recommend`, `POST /api/predict`, and paginated `GET /api/runs?offset=0`; generated OpenAPI at `/docs`.
- At most 1,000 candidates, ten returned results, 32 KiB request bodies, one active inference and 20 inference requests per user per minute per process. Busy workers return 429; the UI preserves retryable errors. Limits reset on restart and are not distributed protection. Add gateway abuse controls and a durable queue before multiple workers or replicas.
- Saved runs use Supabase rather than Render disk. Own-user select/insert are granted; update/delete are intentionally absent from this UI and migration. An administrator needs a retention/deletion policy before customer onboarding. User deletion cascades to owned runs.
- Users can insert their own JSON records through PostgREST; saved reports are personal research records, not tamper-proof certified results. For regulated traceability, add server-signed reports, controlled write paths and an audit ledger.
- No payments, quotas by subscription, enterprise tenancy or new material-family models are implemented. Existing checkpoint defects and research-only evidence remain documented in the model card.
- Keep one deployment revision for backend/frontend compatibility. To roll back, redeploy the previous working API and frontend revisions together. The additive table can remain unused; do not drop it or reset the shared database as a routine rollback.
- Use provider dashboards for error counts, memory, latency and available storage. Never log passwords, bearer tokens or customer compositions unnecessarily. Define backup/restore and retention requirements before storing customer work.

## Current access and verification status

The local Mineral-ML project had templates but no configured cloud links. Vercel CLI was not authenticated; hosting connectors were unavailable; browser automation could not initialize in this session. Docker CLI was installed but its Linux daemon was not running, so the Linux image build and live cloud checks remain unverified. The working local preview uses native Python and Vite. No shared Mineral-ML resources were modified.
