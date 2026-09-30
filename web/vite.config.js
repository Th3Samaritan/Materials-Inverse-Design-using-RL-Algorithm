import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ command, mode }) => {
  const env = { ...loadEnv(mode, process.cwd(), 'VITE_'), ...process.env };
  if (command === 'build') {
    for (const name of ['VITE_API_URL', 'VITE_SUPABASE_URL', 'VITE_SUPABASE_PUBLISHABLE_KEY']) {
      if (!env[name]) throw new Error(`${name} is required for a hosted build.`);
    }
    for (const name of ['VITE_API_URL', 'VITE_SUPABASE_URL']) {
      if (new URL(env[name]).protocol !== 'https:') throw new Error(`${name} must use HTTPS.`);
    }
    const key = env.VITE_SUPABASE_PUBLISHABLE_KEY;
    let role;
    if (key.startsWith('eyJ')) {
      try { role = JSON.parse(Buffer.from(key.split('.')[1], 'base64url').toString()).role; } catch { throw new Error('Invalid Supabase key.'); }
    }
    if (key.startsWith('sb_secret_') || role === 'service_role') throw new Error('Never include a Supabase secret/service-role key in the frontend.');
    if (!key.startsWith('sb_publishable_') && role !== 'anon') throw new Error('Use a Supabase publishable or legacy anon key.');
  }
  return { plugins: [react()], test: { include: ['src/**/*.test.jsx'], environment: 'jsdom', globals: true }, server: { port: 5173, strictPort: true } };
});
