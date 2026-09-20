# Laya in the browser

`packages/laya-web` (`@laya-mlx/web`) runs a Laya ONNX bundle with onnxruntime-web (WebGPU, wasm fallback)
and returns the same JSON as the Python `Agent.predict`. `apps/demo` is the static site: a parity page now,
Snake and Chess demos next.

```bash
pnpm install
pnpm typecheck && pnpm test                      # Node tests; fixture based, no model needed
LAYA_TOKENIZER_DIR=<bundle>/tokenizer pnpm test  # also the tokenizer / prompt / agent parity tests
pnpm --filter laya-demo dev                      # http://localhost:5173
pnpm test:browser                                # builds the site, runs the index smoke test
LAYA_MODEL_URL=https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/ pnpm test:browser
```

The real-model test asserts that the WebGPU provider was used, which currently only works on macOS
(the Playwright config passes `--use-angle=metal` on darwin); on Linux it will fail on that provider
assertion.

`uv run python -m benchmarks.web_fixtures` (repository root) regenerates `fixtures/parity-multilingual.json`
from the MLX runtime; the TypeScript tests compare token ids, marker positions, collated batches and calibrated
answers against it byte for byte. A bundle directory comes from `uv run laya-mlx export-onnx --dtype float16`.

The demo reads the model from `VITE_LAYA_MODEL_URL` (default: the Hugging Face bundle above). For offline
development, symlink a local bundle under `apps/demo/public/models/<name>` (git-ignored) and set
`VITE_LAYA_MODEL_URL=/models/<name>/`; `vite preview` serves that directory too.

Usage:

```ts
import { loadAgent } from "@laya-mlx/web";

const { agent, provider } = await loadAgent(
  "https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/",
  { wasmPaths: "/ort/", onProgress: (p) => console.log(p.received, p.total) },
);
const result = await agent.predict("I was billed twice.", {
  department: {
    type: "choice",
    instructions: "Who should handle this?",
    criteria: ["billing", "sales"],
  },
});
console.log(provider, result.answers.department);
```

`wasmPaths` must point at a directory serving onnxruntime-web's `.wasm`/`.mjs` files; the demo copies them
to `ort/` at build time and serves them from `node_modules` in dev. WebGPU needs a secure context (https or
localhost). onnxruntime-web is pinned to 1.30.0 because its WebGPU backend needs `graphOptimizationLevel:
"basic"` for this graph (see the comment in `packages/laya-web/src/session.ts`).
