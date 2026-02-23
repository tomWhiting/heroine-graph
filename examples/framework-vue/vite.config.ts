import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

// Plugin to handle WGSL files as raw text (core source imports .wgsl directly)
function wgslPlugin() {
  return {
    name: "wgsl-loader",
    transform(_code: string, id: string) {
      if (id.endsWith(".wgsl")) {
        const content = readFileSync(id, "utf-8");
        return {
          code: `export default ${JSON.stringify(content)};`,
          map: null,
        };
      }
    },
  };
}

export default defineConfig({
  plugins: [vue(), wgslPlugin()],
  resolve: {
    // Alias values must be absolute paths: Vite substitutes relative strings
    // as-is and resolves them against the importing module, which breaks for
    // the wrapper's own "@graphmother/core" imports.
    alias: {
      "@graphmother/core": resolve(__dirname, "../../packages/core/mod.ts"),
      "@graphmother/vue": resolve(__dirname, "../../packages/vue/src/index.ts"),
      "@graphmother/wasm": resolve(
        __dirname,
        "../../dist/graphmother_wasm.js",
      ),
    },
  },
  optimizeDeps: {
    exclude: ["@graphmother/wasm"],
  },
  assetsInclude: ["**/*.wasm"],
});
