import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";
import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

const rootDir = fileURLToPath(new URL(".", import.meta.url));
const resolve = (...segments: string[]) =>
  fileURLToPath(new URL(segments.join("/"), import.meta.url));

/** `/ort/<name>` -> the matching content type, or `null` for anything else (including a miss). */
function ortContentType(name: string): string | null {
  if (name.endsWith(".wasm")) return "application/wasm";
  if (name.endsWith(".mjs")) return "text/javascript";
  return null;
}

/**
 * Serves onnxruntime-web's .wasm/.mjs files under `/ort/` during `vite dev`, straight out of
 * `node_modules/onnxruntime-web/dist/` (read from disk on every request; these are dev-only
 * conveniences, not perf-sensitive). `src/model-url.ts` points `wasmPaths` at `${BASE_URL}ort/`
 * unconditionally — the same path prod's `dist/ort/` copy (see `viteStaticCopy` below) serves —
 * so this is what makes that path resolve during `vite dev` too. Deliberately not
 * `vite-plugin-static-copy`'s own dev-serve middleware: that plugin populates its file map
 * asynchronously on `buildStart`, which races a freshly started dev server (a request that
 * lands first falls through to Vite's SPA `index.html` fallback instead of 404ing, so failures
 * were silent); a plain fs read has no such warm-up window.
 */
function ortDevAssetsPlugin(): Plugin {
  const ortDir = resolve("node_modules/onnxruntime-web/dist/");
  return {
    name: "laya-demo:ort-dev-assets",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const base = server.config.base;
        const prefix = `${base}ort/`;
        const url = req.url ?? "";
        if (!url.startsWith(prefix)) return next();
        const name = decodeURIComponent(url.slice(prefix.length).split("?")[0] ?? "");
        const contentType = ortContentType(name);
        if (!name || name.includes("/") || !contentType) {
          res.statusCode = 404;
          res.end();
          return;
        }
        try {
          const data = await readFile(join(ortDir, name));
          res.setHeader("Content-Type", contentType);
          res.end(data);
        } catch {
          res.statusCode = 404;
          res.end();
        }
      });
    },
  };
}

/** `/models/<...path>` -> the matching content type; `application/octet-stream` covers `model.onnx`. */
function modelAssetContentType(name: string): string {
  return name.endsWith(".json") ? "application/json" : "application/octet-stream";
}

/**
 * Serves `public/models/` under `/models/` during `vite preview` only. `vite preview` serves
 * `dist/`, and `build.copyPublicDir: false` (below) deliberately keeps `public/models/` — a
 * symlink to a local ONNX bundle used only for the model-gated e2e test, see its own comment —
 * out of `dist/`. Without this, the LAYA_MODEL_URL=http://127.0.0.1:4173/models/... test run
 * would 404 against a built/previewed site. This is a local-test convenience only: nothing in
 * `src/` references `/models/`, and it is never part of what ships.
 */
function modelsPreviewPlugin(): Plugin {
  const modelsDir = resolve("public/models/");
  return {
    name: "laya-demo:models-preview-only",
    configurePreviewServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const base = server.config.base;
        const prefix = `${base}models/`;
        const url = req.url ?? "";
        if (!url.startsWith(prefix)) return next();
        const rel = normalize(decodeURIComponent(url.slice(prefix.length).split("?")[0] ?? ""));
        const filePath = join(modelsDir, rel);
        if (rel.startsWith("..") || !filePath.startsWith(modelsDir)) return next();
        try {
          const stats = await stat(filePath);
          if (!stats.isFile()) return next();
          res.setHeader("Content-Type", modelAssetContentType(filePath));
          res.setHeader("Content-Length", String(stats.size));
          createReadStream(filePath).pipe(res);
        } catch {
          next();
        }
      });
    },
  };
}

/**
 * onnxruntime-web's JSEP loader has a `new URL("ort-wasm-simd-threaded.jsep.wasm",
 * import.meta.url)`-style fallback that Rollup statically detects and emits as a hashed asset
 * under `dist/assets/` (~27 MB for the .wasm alone). It is never fetched at runtime: `wasmPaths`
 * (see model-url.ts) always points onnxruntime-web at `dist/ort/` instead, which `viteStaticCopy`
 * below populates explicitly — but Rollup has no way to know that from the source, so without
 * this the same files ship twice. Deleting the emitted-fallback bundle entries and having the
 * e2e parity test still pass afterwards is exactly the proof that the fallback path is unused.
 */
function dropOrtWasmFallbackAssetsPlugin(): Plugin {
  return {
    name: "laya-demo:drop-ort-wasm-fallback-assets",
    generateBundle(_options, bundle) {
      for (const fileName of Object.keys(bundle)) {
        if (/^assets\/ort-wasm.*\.(wasm|mjs)$/.test(fileName)) delete bundle[fileName];
      }
    },
  };
}

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
    // `public/models/` holds a symlink to a local ONNX bundle used only for manual/e2e testing;
    // the directory is git-ignored and documented in `web/README.md`. Vite's default publicDir
    // copy would dereference that symlink and duplicate the 600+ MB bundle into `dist/` on every
    // build, so publicDir copying is disabled and only what the site actually needs (the ORT
    // runtime assets, the parity fixture) is copied explicitly below.
    copyPublicDir: false,
  },
  plugins: [
    // Serve plugin filtered out (`.filter`): dev/preview serving of both targets below is
    // handled explicitly instead (`ortDevAssetsPlugin`, and public/fixtures/parity-*.json is
    // Vite's ordinary `public/` serving) — see `ortDevAssetsPlugin`'s comment for why.
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
          // The fixture also lives as a committed symlink at `public/fixtures/parity-
          // multilingual.json` (-> `../../../../fixtures/parity-multilingual.json`), which is
          // what `vite dev`/`vite preview` serve directly from `public/`. This target is what
          // the production `build` copies instead, since `copyPublicDir: false` (above) means
          // `public/` is not copied wholesale. If the fixture ever moves, both of these need to
          // point at the new location.
          src: "../../fixtures/parity-multilingual.json",
          dest: "fixtures",
          rename: { stripBase: true },
        },
      ],
    }).filter((plugin) => plugin.apply !== "serve"),
    ortDevAssetsPlugin(),
    modelsPreviewPlugin(),
    dropOrtWasmFallbackAssetsPlugin(),
  ],
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
