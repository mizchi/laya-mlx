# Laya in the browser

`packages/laya-web` (`@laya-mlx/web`) runs a Laya ONNX bundle with onnxruntime-web (WebGPU, wasm fallback)
and returns the same JSON as the Python `Agent.predict`. `apps/demo` is the static site with three pages:

- `index.html` — landing page
- `parity.html` — fixture-driven parity check against the MLX runtime
- `snake.html` — the browser Snake demo (see [docs/SNAKE_DEMO.md](../docs/SNAKE_DEMO.md) for URL parameters
  and controls)

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

## Deploying

Two GitHub Actions workflows publish `apps/demo/dist`, independently:

- [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) builds with
  `VITE_BASE=/<repo-name>/` and deploys to GitHub Pages on every push to `main`. GitHub Pages must be set to
  **Source: GitHub Actions** (repository Settings → Pages) before this workflow can deploy; it needs no
  secrets.
- [`.github/workflows/space.yml`](../.github/workflows/space.yml) builds with the default base (`/`), copies
  [`space/README.md`](space/README.md) in as `dist/README.md` (the Space's model card / front matter), and
  `hf upload`s the result to a Hugging Face static Space on push to `main`. It is gated on the repository
  **variable** `HF_SPACE` (e.g. `mizchi/laya-web-demo`) being set — the job is skipped entirely when it is
  empty — and needs the repository **secret** `HF_TOKEN` (a Hugging Face token with write access to that
  Space).

`space/README.md` carries the Space's front matter (`title`, `emoji`, `colorFrom`/`colorTo`, `sdk: static`,
`license`, `models:`) plus a short description; Hugging Face renders it as the Space's landing page alongside
the app. Edit it the same way as any other model-card front matter.

Both workflows can also be triggered manually via `workflow_dispatch`.
