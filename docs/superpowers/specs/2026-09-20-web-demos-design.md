# Browser runtime and demos (Snake, Chess) — design

Date: 2026-09-20. Status: approved for planning.

## Goal

Run Laya decision models in the browser from the ONNX export (`laya-mlx export-onnx`, see `docs/ONNX_EXPORT.md`) and ship two demos as a static site:

- **Snake**: the existing terminal demo, self-playing, ported to a canvas page.
- **Chess**: the user plays against Laya; a shallow planner proposes candidate moves and Laya chooses.

The published site downloads the model from Hugging Face. The Python package is not changed except for one fixture-dump script.

## Decisions already made

| Topic | Decision |
|---|---|
| Snake interaction | AI self-play only (same as the terminal demo). No two-snake mode. |
| Model precision in the browser | float16 bundle (647 MB). WebGPU float16 keeps 63/63 selected answers; probability error 1.2e-2 is invisible in the UI. |
| Model hosting | Hugging Face model repo, fetched directly via `resolve/` URLs (CORS enabled). |
| Site hosting | GitHub Pages from this repository. |
| Code location | `web/` pnpm workspace in this repository. |
| UI stack | Vite + TypeScript, no framework. Snake on canvas, Chess as a DOM grid. |
| Chess strength | Candidate shortlist from a 2-ply material search, Laya picks among ≤6 candidates. Target: beats a beginner. |
| Execution providers | `webgpu`, falling back to `wasm` (single-threaded; GitHub Pages cannot set COOP/COEP). WebGL is not supported. |

## Layout

```
web/
  package.json, pnpm-workspace.yaml, tsconfig.base.json
  packages/laya-web/        @laya-mlx/web — runtime library
  apps/demo/                Vite site: /snake, /chess
  fixtures/                 JSON dumped by benchmarks/web_fixtures.py (committed)
benchmarks/web_fixtures.py  new: dumps parity fixtures for the JS tests
.github/workflows/pages.yml new: builds web/apps/demo and deploys to GitHub Pages
```

## 1. Runtime library `@laya-mlx/web`

Four modules with one clear purpose each. Everything except `session.ts` is pure and testable in Node.

| Module | Depends on | Responsibility | Python counterpart |
|---|---|---|---|
| `types.ts` | – | Question and answer schema as TypeScript types (the contract). | `Agent._to_internal` validation, result shape |
| `prompt.ts` | tokenizer | `renderOptions`, `buildPrefix`, `buildSequence`, `collate`. Byte-for-byte the same token ids and marker positions as Python, including mask-token scrubbing, the 48-token option cap, the head budget rules and truncation. | `common.py`, `agent.collate_items` |
| `calibration.ts` | – | temperature buckets, softmax, entropy confidence, expected score, answer objects with four-decimal rounding, `usage`. | `common.py`, `Agent.system_one` |
| `session.ts` | onnxruntime-web | Download the bundle with progress, store `model.onnx` in the Cache API, create the `InferenceSession` (`webgpu` → `wasm`), `run(batch)` → `{logits, actLogits}` as Float32Array. | `Agent.forward` |
| `agent.ts` | all above | `LayaAgent.load(bundleUrl, options)` and `predict(state, questions)`. Batches in chunks of 16, at least two marker slots per row, pads like Python. Returns the same JSON as the Python `predict`. | `Agent` |

Tokenizer: `@huggingface/tokenizers` loading `tokenizer.json`; if it cannot reproduce Python ids for the fixtures, use the tokenizer from Transformers.js instead. The decision is made by the first test below.

Bundle URL convention: a directory URL; the runtime appends `model.onnx`, `rl_agent_config.json`, `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`, `onnx_config.json`. Works for `https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/` and for a local `models/` directory served by Vite.

Cache: `caches.open("laya-models")` keyed by the model URL. Any failure (quota, private mode) falls back to a plain fetch without surfacing an error. The loading UI reports bytes received from `Content-Length` and the response stream.

Memory note: the model is fetched into an ArrayBuffer and handed to onnxruntime, so the tab needs roughly twice the file size. The demos are documented as desktop-only.

### Tests

1. `prompt.test.ts` (vitest, Node): for all 63 fixture questions, `ids`, `markers` and `qtype` equal the Python dump.
2. `calibration.test.ts` (vitest): feeding the Python logits produces `answers` and `usage` identical to the Python `predict()` output.
3. `session.spec.ts` (Playwright, Chromium with WebGPU): with a real model, 63/63 argmax agreement and probability error < 2e-2 against the MLX CPU float32 logits. Runs only when `LAYA_MODEL_URL` is set; CI runs the Node tests and a stub-agent smoke only.

## 2. Snake demo

Direct port of `laya_mlx/snake/game.py` and the compact prompt of `policy.py`. The strings sent to Laya are identical to Python and are checked against fixtures.

```
apps/demo/src/snake/
  game.ts    rules, Hamiltonian cycle, legal/safe moves with advance count, food BFS (pure)
  policy.ts  board → state string + three questions (move choice, risk noul, food noul); safety shield
  loop.ts    pacing (default 12 decisions/s), max-speed mode, pause; decisions are awaited sequentially
  view.ts    canvas board + side panel: probability bars, EXECUTING, DEAD-END RISK, FOOD REACHABLE,
             INFERENCE ms, DECISIONS/s, OUTPUT TOKENS 0, ENGINE (provider · dtype), Shield interventions
  main.ts    wiring only
```

- Controls: Space pause, ↑/↓ speed by 2 decisions/s, R new round (next seed), a max-speed toggle button.
- RNG: seeded mulberry32. Board sequences are not reproducible against Python runs; rule and prompt equivalence are.
- Shield: as in Python, if the model's argmax is not a safe direction, execute the safe direction with the highest probability and count an intervention.

Tests: `game.test.ts` (cycle validity, legal/safe classification, BFS reachability, win/death), `policy.test.ts` (Python-dumped boards produce identical `state` and `questions`), Playwright smoke with a stub agent (board renders, pause and reset work).

## 3. Chess demo

```
apps/demo/src/chess/
  candidates.ts  legal moves (chess.js) → 2-ply material search → top 6 with descriptions (pure)
  policy.ts      candidates → Laya choice question + "king in danger" noul; safety shield
  board.ts       DOM 8×8, click to move with legal-target highlights, promotion always to queen
  panel.ts       candidate probability bars, EXECUTING, shield count, inference ms, SAN move list, status
  main.ts        wiring, new game, side switch
```

Candidate generation: for each legal move, evaluate after the opponent's best material reply. Score = material difference + small bonuses (check, castling, developing a minor piece, center pawn), checkmate = ±∞, stalemate = 0. Keep the best six. Descriptions are templates, the best-scoring candidate gets `Best.`:

`Checkmate. Best.` · `Captures the queen. Safe.` · `Hangs the knight to a pawn.` · `Gives check. Safe.` · `Castles kingside. Safe.` · `Develops the bishop. Safe.` · `Quiet move. Safe.`

Laya questions: one `choice` (`Choose the strongest safe move.`, criteria = SAN → description) and one `noul` (`Is our king in danger?`) for the panel. About 50 ms per move on WebGPU.

Shield: if the chosen candidate scores at least 3 material points below the best candidate, or allows mate in one when another candidate does not, play the best candidate and count an intervention. Shown in the panel exactly like the Snake shield.

Game flow: user is white by default, switchable. Game end (checkmate, stalemate, 50 moves, threefold repetition, insufficient material) comes from chess.js. No undo.

Tests: `candidates.test.ts` (finds mate in one, flags a hanging piece, description strings, ordering), `policy.test.ts` (shield conditions), Playwright: the user makes a move and the AI replies (stub agent in CI; real model when `LAYA_MODEL_URL` is set).

## 4. Distribution and deployment

- Upload the float16 bundle with `hf upload mizchi/laya-multilingual-onnx <bundle dir>` including a model card that records the source checkpoint, revision, export command, opset and the parity report. Owner: `mizchi` (login verified on 2026-09-20).
- Demo model URL is a build-time setting (`VITE_LAYA_MODEL_URL`), defaulting to the Hugging Face URL; local development points it at `models/`.
- `pages.yml`: on push to `main`, `pnpm install`, `pnpm -C web build`, deploy `web/apps/demo/dist` to GitHub Pages. onnxruntime-web assets are copied into `dist` so the site does not depend on a CDN.
- CI (`ci.yml`): add a job for `pnpm -C web lint`, `typecheck`, `test` (Node tests and stub-agent Playwright).

## Out of scope

Two-snake mode, chess undo, mobile support, int8/int4 models, English (421M) checkpoints, the Python runtime.
