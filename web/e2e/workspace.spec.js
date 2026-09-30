import { expect, test } from '@playwright/test';

test('real model search, export, invalidation and forward prediction', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Find candidate processes' })).toBeVisible();
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(255, 255, 255)');
  await page.screenshot({ path: '../.cache/workspace-desktop.png', fullPage: true });
  await page.getByRole('button', { name: 'Find candidate processes' }).click();
  await expect(page.getByText('4 candidates to explore')).toBeVisible();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export research run' }).click();
  expect((await download).suggestedFilename()).toBe('mast-research-run.json');
  await page.screenshot({ path: '../.cache/workspace-results.png', fullPage: true });
  await page.getByLabel('Tensile strength (MPa)').fill('1400');
  await expect(page.getByText('4 candidates to explore')).toHaveCount(0);
  await page.getByRole('button', { name: 'Predict properties', exact: true }).click();
  await page.getByRole('button', { name: 'Predict properties', exact: true }).last().click();
  await expect(page.getByRole('heading', { name: 'Predicted properties' })).toBeVisible();
});

test('mobile layout fits the viewport and planned families cannot run', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Find candidate processes' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: '../.cache/workspace-mobile.png', fullPage: true });
  await page.getByLabel('Material family').selectOption('Elastomers');
  await expect(page.getByRole('button', { name: 'Find candidate processes' })).toHaveCount(0);
  await expect(page.getByText(/Predictions are not yet available/)).toBeVisible();
});
