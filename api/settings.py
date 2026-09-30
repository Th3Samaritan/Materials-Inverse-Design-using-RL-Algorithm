"""Environment configuration; production never permits anonymous inference."""
import os
import json
import base64
from urllib.parse import urlparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    environment: str = 'production'
    anonymous: bool = False
    supabase_url: str = ''
    supabase_key: str = ''
    origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls):
        settings = cls(
            environment=os.getenv('APP_ENV', 'production'),
            anonymous=os.getenv('ALLOW_ANONYMOUS', 'false').lower() == 'true',
            supabase_url=os.getenv('SUPABASE_URL', '').rstrip('/'),
            supabase_key=os.getenv('SUPABASE_PUBLISHABLE_KEY', ''),
            origins=tuple(s.strip() for s in os.getenv('CORS_ORIGINS', 'http://localhost:5173').split(',') if s.strip()),
        )
        if settings.anonymous and (settings.environment != 'development' or os.getenv('RENDER')):
            raise RuntimeError('Anonymous mode is restricted to local development.')
        if not settings.anonymous and (not settings.supabase_url or not settings.supabase_key):
            raise RuntimeError('SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY are required.')
        if not settings.anonymous:
            if urlparse(settings.supabase_url).scheme != 'https':
                raise RuntimeError('Hosted Supabase must use HTTPS.')
            key = settings.supabase_key
            role = None
            if key.startswith('eyJ'):
                try:
                    encoded = key.split('.')[1]
                    role = json.loads(base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)))['role']
                except (ValueError, KeyError, IndexError):
                    raise RuntimeError('Invalid Supabase publishable key.') from None
            if not key.startswith('sb_publishable_') and role != 'anon':
                raise RuntimeError('Use a publishable or legacy anon key, never a secret/service-role key.')
        if '*' in settings.origins:
            raise RuntimeError('Use explicit CORS origins.')
        return settings
