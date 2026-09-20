/** Directory URL of the ONNX bundle; override with VITE_LAYA_MODEL_URL (for example /models/laya-multilingual-onnx-fp16/). */
export const MODEL_URL: string =
  import.meta.env.VITE_LAYA_MODEL_URL ??
  "https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/";

/**
 * Where onnxruntime-web finds its .wasm/.mjs assets. The same path is valid in both dev and
 * prod: `vite-plugin-static-copy` copies the files into `dist/ort/` at build time (see
 * vite.config.ts) for prod, and a `configureServer` middleware there (`ortDevAssetsPlugin`)
 * serves the same files from `node_modules/onnxruntime-web/dist/` under `/ort/` during
 * `vite dev`, so there is no dev/prod branch here.
 */
export const ORT_WASM_PATHS: string = `${import.meta.env.BASE_URL}ort/`;
