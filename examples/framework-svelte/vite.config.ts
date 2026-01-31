import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  resolve: {
    alias: {
      "@heroine-graph/core": "../../packages/core/mod.ts",
      "@heroine-graph/svelte": "../../packages/svelte/src/index.ts",
    },
  },
});
