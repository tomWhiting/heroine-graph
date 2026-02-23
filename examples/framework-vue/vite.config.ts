import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@graphmother/core": "../../packages/core/mod.ts",
      "@graphmother/vue": "../../packages/vue/src/index.ts",
    },
  },
});
