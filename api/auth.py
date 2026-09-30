"""Validate access tokens with Supabase Auth, never trust client user IDs."""
from dataclasses import dataclass
import httpx
from fastapi import HTTPException, Request


@dataclass(frozen=True)
class Identity:
    user_id: str
    token: str


async def identity(request: Request):
    settings = request.app.state.settings
    header = request.headers.get('authorization', '')
    if not header and settings.anonymous:
        return Identity('local-development', '')
    if not header.startswith('Bearer ') or len(header) > 8192:
        raise HTTPException(401, 'Sign in to run material predictions.')
    token = header[7:]
    try:
        response = await request.app.state.http.get(
            f'{settings.supabase_url}/auth/v1/user',
            headers={'apikey': settings.supabase_key, 'Authorization': f'Bearer {token}'},
        )
        if response.status_code in (401, 403):
            raise HTTPException(401, 'Your session has expired. Please sign in again.')
        response.raise_for_status()
        data = response.json()
        if not isinstance(data.get('id'), str):
            raise ValueError('Missing identity')
        return Identity(data['id'], token)
    except (httpx.HTTPError, ValueError, KeyError):
        raise HTTPException(503, 'Authentication service is unavailable.') from None


async def store_run(request, user, report):
    settings = request.app.state.settings
    if not user.token:
        raise HTTPException(409, 'Saving requires a Supabase account; local anonymous runs can be exported.')
    response = await request.app.state.http.post(
        f'{settings.supabase_url}/rest/v1/mast_runs',
        headers={'apikey': settings.supabase_key, 'Authorization': f'Bearer {user.token}', 'Prefer': 'return=representation'},
        json={'user_id': user.user_id, 'report': report},
    )
    response.raise_for_status()
    return response.json()[0]['id']
