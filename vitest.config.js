import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vitest/config';

import { storybookTest } from '@storybook/addon-vitest/vitest-plugin';

import { playwright } from '@vitest/browser-playwright';

const dirname =
  typeof __dirname !== 'undefined' ? __dirname : path.dirname(fileURLToPath(import.meta.url));

// More info at: https://storybook.js.org/docs/next/writing-tests/integrations/vitest-addon
export default defineConfig({
  test: {
    projects: [
      {
        extends: true,
        plugins: [
          // The plugin will run tests for the stories defined in your Storybook config
          // See options at: https://storybook.js.org/docs/next/writing-tests/integrations/vitest-addon#storybooktest
          storybookTest({ configDir: path.join(dirname, '.storybook') }),
        ],
        test: {
          name: 'storybook',
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({
              launchOptions: {
                // Headless Chromium ships without WebGPU unless explicitly
                // enabled; without these flags every story falls into the
                // "WebGPU not supported" error path and the suite silently
                // stops covering the renderer.
                args: [
                  '--enable-unsafe-webgpu',
                  '--enable-gpu',
                  // Vulkan backend for Linux CI runners; ignored on macOS
                  // (which uses Metal via ANGLE) and Windows (D3D).
                  '--enable-features=Vulkan',
                ],
              },
            }),
            instances: [{ browser: 'chromium' }],
          },
          setupFiles: ['.storybook/vitest.setup.js'],
        },
      },
    ],
  },
});
