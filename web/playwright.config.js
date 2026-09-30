import { defineConfig } from '@playwright/test';

// Run against the local API and Vite servers described in the root README.
export default defineConfig({
  testDir: './e2e',
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:5173', channel: process.env.PLAYWRIGHT_CHANNEL || undefined },
  outputDir: '../.cache/browser-tests',
});
