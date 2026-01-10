import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

export default {
  resolve: {
    alias: {
      '@heroine-graph/wasm': resolve(__dirname, '../../dist/heroine_graph_wasm.js'),
    },
  },
  optimizeDeps: {
    exclude: ['@heroine-graph/wasm'],
  },
  assetsInclude: ['**/*.wasm'],
};
