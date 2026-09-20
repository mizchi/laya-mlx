import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

const rootDir = fileURLToPath(new URL(".", import.meta.url));
const resolve = (...segments: string[]) =>
  fileURLToPath(new URL(segments.join("/"), import.meta.url));

// onnxruntime-web loads its wasm/worker files by URL at runtime; ship them next to the pages.
export default defineConfig({
  root: rootDir,
  base: process.env.VITE_BASE ?? "/",
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        index: resolve("index.html"),
        parity: resolve("parity.html"),
      },
    },
    // `public/models/` holds a symlink to a local ONNX bundle used only for manual/e2e testing
    // (see public/models/.gitkeep-equivalent note in README-less form: it's gitignored). Vite's
    // default publicDir copy would dereference that symlink and duplicate the 600+ MB bundle
    // into `dist/` on every build, so publicDir copying is disabled and only what the site
    // actually needs (the ORT runtime assets, the parity fixture) is copied explicitly below.
    copyPublicDir: false,
  },
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/onnxruntime-web/dist/*.{wasm,mjs}",
          dest: "ort",
          // Without this, v4 preserves each matched file's full path (including the
          // `node_modules/onnxruntime-web/dist/` prefix) under `dest`; flatten to `dist/ort/*`.
          rename: { stripBase: true },
        },
        {
          src: "../../fixtures/parity-multilingual.json",
          dest: "fixtures",
          rename: { stripBase: true },
        },
      ],
    }),
  ],
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
