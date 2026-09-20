import { loadAgent, type LoadedAgent } from "@laya-mlx/web";

import { MODEL_URL, ORT_WASM_PATHS } from "./model-url.ts";

export interface LoaderUi {
  status(text: string): void;
}

/** Loads the bundle named by `?model=` (or the default) with progress text; `stub` is handled by callers. */
export async function loadDemoAgent(ui: LoaderUi, batchSize = 3): Promise<LoadedAgent> {
  const url = new URLSearchParams(location.search).get("model") ?? MODEL_URL;
  ui.status(`fetching model from ${url}`);
  const loaded = await loadAgent(url, {
    wasmPaths: ORT_WASM_PATHS,
    batchSize,
    onProgress: (p) => {
      const mb = (n: number) => (n / 1e6).toFixed(0);
      ui.status(
        p.total
          ? `downloading model ${mb(p.received)} / ${mb(p.total)} MB`
          : `downloading model ${mb(p.received)} MB`,
      );
    },
  });
  ui.status(`ready on ${loaded.provider} · ${loaded.bundle.onnxConfig.dtype}`);
  return loaded;
}
