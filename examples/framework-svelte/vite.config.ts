import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  resolve: {
    alias: {
      "@graphmother/core": "../../packages/core/mod.ts",
      "@graphmother/svelte": "../../packages/svelte/src/index.ts",
    },
  },
});
