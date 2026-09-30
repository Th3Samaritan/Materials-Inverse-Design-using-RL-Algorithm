"""Render-hosted FastAPI service. Vercel serves the separate frontend."""
from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore
import os
import time
import httpx
import torch
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from mast.catalog import ELEMENT_NAMES, PROPERTY_NAMES, PROP_UNITS, COMP_RANGES, PROP_RANGES, STEEL_PRESETS, QUENCH_MAP
from mast.engine import FAMILIES, Constraints, load_models, recommend, validate_composition, candidate_mask
from mast.models import compute_ac3_torch
from .settings import Settings
from .schemas import DesignRequest, ForwardRequest
from .auth import Identity, identity, store_run

ROOT = Path(__file__).resolve().parents[1]


def create_app(settings=None, models=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(int(os.getenv('TORCH_NUM_THREADS', '2')))
        app.state.models = models or load_models(ROOT/'checkpoints/forward_model.pt', ROOT/'checkpoints/policy.pt')
        async with httpx.AsyncClient(timeout=15) as client:
            app.state.http = client
            yield

    app = FastAPI(title='MAST Materials API', version='1.0.0', lifespan=lifespan)
    app.state.settings = settings
    app.state.slots = BoundedSemaphore(1)
    app.state.rates = {}
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.origins),
                       allow_methods=['GET', 'POST'], allow_headers=['Authorization', 'Content-Type'])

    @app.middleware('http')
    async def size_and_headers(request, call_next):
        # Stream and bound JSON bodies even when Content-Length is omitted.
        if request.method == 'POST':
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 32768:
                    from fastapi.responses import JSONResponse
                    return JSONResponse({'detail':'Request body exceeds 32 KiB.'}, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    async def run_limited(user, function):
        now = time.monotonic()
        app.state.rates = {k:v for k,v in app.state.rates.items() if now-v[0] < 60}
        start, count = app.state.rates.get(user.user_id, (now, 0))
        if count >= 20:
            raise HTTPException(429, 'Please wait a minute before running more searches.', headers={'Retry-After':'60'})
        if not app.state.slots.acquire(blocking=False):
            raise HTTPException(429, 'The inference worker is busy. Try again shortly.', headers={'Retry-After':'3'})
        app.state.rates[user.user_id] = (start, count+1)
        def worker():
            try:
                return function()
            finally:
                app.state.slots.release()
        try:
            return await run_in_threadpool(worker)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get('/health')
    async def health():
        return {'status':'ok', 'model_ready':hasattr(app.state,'models'), 'version':'1.0.0'}

    @app.get('/api/catalog')
    async def catalog():
        return dict(elements=ELEMENT_NAMES, properties=PROPERTY_NAMES, units=PROP_UNITS,
                    composition_ranges=COMP_RANGES, property_ranges=PROP_RANGES,
                    presets={k:v for k,v in STEEL_PRESETS.items() if k != 'H13 Tool'},
                    families=FAMILIES, quenches=QUENCH_MAP, auth_required=not settings.anonymous)

    @app.post('/api/recommend')
    async def design(body: DesignRequest, request: Request, user: Identity = Depends(identity)):
        values = body.model_dump(exclude={'save','composition','targets','constraints'})
        equipment = Constraints(**{**body.constraints.model_dump(), 'allowed_quenches':tuple(body.constraints.allowed_quenches)})
        report = await run_limited(user, lambda: recommend(app.state.models, body.composition, body.targets, constraints=equipment, **values))
        persistence = {'status':'not_requested'}
        if body.save:
            try:
                run_id = await store_run(request, user, report)
                persistence = {'status':'saved', 'id':run_id}
            except (httpx.HTTPError, HTTPException, ValueError, KeyError, IndexError):
                persistence = {'status':'failed', 'message':'Prediction completed, but saving failed. Export this run to keep a copy.'}
        return {'report':report, 'persistence':persistence}

    @app.post('/api/predict')
    async def predict(body: ForwardRequest, user: Identity = Depends(identity)):
        def compute():
            validate_composition(body.composition)
            equipment = Constraints(**{**body.constraints.model_dump(), 'allowed_quenches':tuple(body.constraints.allowed_quenches)})
            c = torch.tensor([[body.composition[e] for e in ELEMENT_NAMES]], dtype=torch.float32)
            p = torch.tensor([[body.process.T_aus, body.process.t_aus, body.process.quench, body.process.T_temper, body.process.t_temper]])
            with torch.inference_mode():
                mean, variance = app.state.models[0](c,p)
            if not candidate_mask(p.numpy(),mean.numpy(),variance.numpy(),compute_ac3_torch(c)[0].item(),equipment)[0]:
                raise ValueError('The process or prediction falls outside supported physical or equipment limits.')
            return dict(pred=dict(zip(PROPERTY_NAMES,mean[0].tolist())), std=dict(zip(PROPERTY_NAMES,variance[0].sqrt().tolist())),
                        evidence_status='synthetic_research_only', model_hashes=app.state.models[-1])
        return await run_limited(user,compute)

    @app.get('/api/runs')
    async def history(request: Request, user: Identity = Depends(identity), offset: int = Query(0,ge=0,le=10000)):
        if not user.token:
            return {'runs':[], 'message':'Sign in to access saved runs.'}
        try:
            response = await app.state.http.get(f'{settings.supabase_url}/rest/v1/mast_runs',
                headers={'apikey':settings.supabase_key, 'Authorization':f'Bearer {user.token}'},
                params={'select':'id,created_at,report', 'order':'created_at.desc', 'limit':'20', 'offset':str(offset)})
            response.raise_for_status()
            return {'runs':response.json()}
        except (httpx.HTTPError, ValueError):
            raise HTTPException(503,'Saved runs are currently unavailable.') from None

    return app
