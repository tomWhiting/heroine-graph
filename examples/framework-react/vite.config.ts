import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@heroine-graph/core": "../../packages/core/mod.ts",
      "@heroine-graph/react": "../../packages/react/src/index.ts",
    },
  },
});
