import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@heroine-graph/core": "../../packages/core/mod.ts",
      "@heroine-graph/vue": "../../packages/vue/src/index.ts",
    },
  },
});
