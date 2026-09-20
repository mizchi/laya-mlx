/** Directory URL of the ONNX bundle; override with VITE_LAYA_MODEL_URL (for example /models/laya-multilingual-onnx-fp16/). */
export const MODEL_URL: string =
  import.meta.env.VITE_LAYA_MODEL_URL ??
  "https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/";

/**
 * Where onnxruntime-web finds its .wasm/.mjs assets.
 *
 * In prod, `vite-plugin-static-copy` copies them into `dist/ort/` at build time (see
 * vite.config.ts) and they are served alongside the page. In dev, that plugin's dev-serve
 * middleware populates its file map asynchronously on `buildStart`, which races a freshly
 * started dev server (a request that lands before the map is ready falls through to Vite's
 * SPA `index.html` fallback instead of 404ing, so failures are silent) — so in dev we instead
 * serve the same files straight out of `node_modules`, which Vite always serves without delay.
 */
export const ORT_WASM_PATHS: string = import.meta.env.DEV
  ? "/node_modules/onnxruntime-web/dist/"
  : `${import.meta.env.BASE_URL}ort/`;
