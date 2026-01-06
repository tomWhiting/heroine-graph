import { resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

/** @type { import('@storybook/html-vite').StorybookConfig } */
const config = {
  "stories": [
    "../stories/**/*.mdx",
    "../stories/**/*.stories.@(js|jsx|mjs|ts|tsx)"
  ],
  "addons": [
    "@chromatic-com/storybook",
    "@storybook/addon-vitest",
    "@storybook/addon-a11y",
    "@storybook/addon-docs"
  ],
  "framework": "@storybook/html-vite",
  async viteFinal(config) {
    return {
      ...config,
      resolve: {
        ...config.resolve,
        alias: {
          ...config.resolve?.alias,
          '@heroine-graph/wasm': resolve(__dirname, '../dist/heroine_graph_wasm.js'),
        },
      },
      optimizeDeps: {
        ...config.optimizeDeps,
        exclude: [...(config.optimizeDeps?.exclude || []), '@heroine-graph/wasm'],
      },
      assetsInclude: [...(config.assetsInclude || []), '**/*.wasm'],
    };
  },
};
export default config;