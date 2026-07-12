import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

// Plugin to handle WGSL files as raw text
function wgslPlugin() {
  return {
    name: "wgsl-loader",
    transform(_code, id) {
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

export default {
  plugins: [wgslPlugin()],
  // Serve static assets (fonts for labels) from dist directory
  // This makes /assets/fonts/roboto-msdf.json available
  publicDir: resolve(__dirname, "../../dist"),
  resolve: {
    alias: {
      "@graphmother/wasm": resolve(
        __dirname,
        "../../dist/heroine_graph_wasm.js",
      ),
    },
  },
  optimizeDeps: {
    exclude: ["@graphmother/wasm"],
  },
  assetsInclude: ["**/*.wasm"],
};
