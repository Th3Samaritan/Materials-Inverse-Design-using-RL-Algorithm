import { createClient } from '@supabase/supabase-js';

export const apiBase = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const url = import.meta.env.VITE_SUPABASE_URL;
const key = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;
export const supabase = url && key ? createClient(url, key) : null;

export async function api(path, { token, signal, ...options } = {}) {
  let response;
  try {
    response = await fetch(`${apiBase}${path}`, {
      ...options, signal: signal || AbortSignal.timeout(90000),
      headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('The inference service could not be reached. It may be starting up; try again shortly.');
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(e => `${e.loc.slice(1).join('.')}: ${e.msg}`).join('; ') : 'The request could not be completed.');
  }
  return data;
}

export function exportRun(report) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = 'mast-research-run.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
