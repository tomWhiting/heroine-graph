import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@graphmother/core": "../../packages/core/mod.ts",
      "@graphmother/react": "../../packages/react/src/index.ts",
    },
  },
});
