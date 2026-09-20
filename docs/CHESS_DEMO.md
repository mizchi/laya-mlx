# Laya Chess: browser demo

Play chess against Laya in the browser, no install required. This is a **feature-assisted
decision demo**, in the same spirit as the [Snake demo](SNAKE_DEMO.md): Laya has no chess
knowledge of its own. A deterministic planner does the actual chess reasoning and hands Laya a
short, described shortlist to pick from.

- Hugging Face Space: https://huggingface.co/spaces/mizchi/laya-web-demo (direct page:
  https://mizchi-laya-web-demo.static.hf.space/chess.html)
- The repository's GitHub Pages site (for the `mizchi/laya-mlx` fork:
  https://mizchi.github.io/laya-mlx/, once Pages is enabled in the repository settings), at `chess.html`

## What Laya does — and does not — do

Laya has no chess training and no board-representation knowledge. All chess reasoning —
legality, material counting, tactics — comes from [chess.js](https://github.com/jhlywa/chess.js)
and a deterministic planner ported alongside it, not from the model.

The planner (`web/apps/demo/src/chess/candidates.ts`):

1. Enumerates every legal move with chess.js.
2. Scores each with a 2-ply material search: play the candidate, then the opponent's best
   (worst-for-us) reply, with a one-step recapture check on the reply's destination square so a
   piece that lands on a defended square isn't misread as hanging (the standard horizon-effect
   fix for a 2-ply search).
3. Adds small positional nudges to break material ties — castling, developing a back-rank
   knight/bishop, occupying a center pawn square, giving check — each worth less than a pawn.
4. Shortlists the top **six** moves by score and describes each in the same plain-English
   template style as Snake: `"Captures the queen. Gives check. Safe. Best."`,
   `"Hangs the knight to a pawn. Loses material."`, `"Allows checkmate. Unsafe."`,
   `"Develops the bishop. Safe."`, `"Quiet move. Safe."` The planner's own top pick is marked
   `Best.` and starred in the panel.

Laya answers exactly one `choice` question over those six descriptions
(`"Choose the strongest safe move."`) and one `noul` question
(`"Is our king in danger?"`, shown as the KING IN DANGER meter). Laya never sees the board
directly — only the state summary and the shortlist's descriptions — so most of the actual chess
knowledge lives in the descriptions the planner writes, not in the model.

## The blunder shield

Same execution-safety pattern as Snake's cycle shield, adapted to chess
(`web/apps/demo/src/chess/policy.ts`). By default (no `unassisted` param) the shield overrides
Laya's raw pick and instead executes the planner's own best candidate when either:

- the picked move scores **at least 3 pawns worse** than the planner's best move (`SHIELD_MARGIN`
  in `policy.ts` — roughly "a beginner would wince"), or
- the picked move allows a mate in one and the planner's best move does not.

When the shield overrides, the panel shows `SHIELD` next to EXECUTING and the intervention
counter increments. `unassisted` disables the shield entirely — Laya's raw first choice is always
played, even if it hangs a piece or walks into mate.

## Strength expectations

This beats a beginner and loses to any real chess engine. The planner's 2-ply material search
with recapture quiescence catches obvious blunders and one-move tactics but has no positional
understanding, no deep tactics, and no opening or endgame knowledge. In the opening most candidates tie on material, so the planner's "Best" mark is then just the first move in generation order and carries no positional signal. The English descriptions
handed to Laya — not Laya's own board reasoning — carry most of the chess knowledge in this demo;
Laya's job is choosing among six pre-scored, pre-described options, not evaluating the position
itself.

## URL parameters

| Parameter | Meaning | Default |
|---|---|---|
| `model` | Model bundle URL, or `stub` for a scripted agent that needs no download/WebGPU | Hugging Face bundle (`mizchi/laya-multilingual-onnx`) |
| `side` | Play as `w` (white) or `b` (black) | `w` |
| `unassisted` | Present (any value) to disable the blunder shield | off (shield on) |
| `fen` | Start from a given FEN instead of the standard opening position | standard opening position |

Example: `chess.html?model=stub&side=b` starts a scripted opponent with the user playing black.
`chess.html?fen=...` (URL-encoded) is useful for testing specific positions, including
checkmate-in-one setups.

## Controls

Click a piece to select it — its legal target squares highlight — then click a highlighted
square to move. Promotion is always a queen; there is no promotion picker. **New game** resets
the board (keeping the current `fen`/`side`); **Play as black** / **Play as white** flips sides
and starts a new game with the AI opening if it now moves first.

## Panel fields

- **CANDIDATES** — up to six shortlisted moves, each with its model probability bar; the ★ marks
  the planner's own best move; the highlighted row is Laya's raw pick before the shield.
- **EXECUTING** — the move actually played, plus a `SHIELD` badge when the blunder shield
  overrode Laya's pick.
- **KING IN DANGER** — Laya's `noul` answer to "Is our king in danger?", as a 0–1 meter.
- **INFERENCE** — measured `Agent.predict` wall time for the one `choice` + one `noul` question
  pair.
- **PLANNER** — time spent generating and describing the candidate shortlist (chess.js move
  generation, 2-ply scoring, and templated descriptions), separate from model inference.
- **ENGINE** — inference backend and precision, e.g. `WebGPU · FP16`, or `STUB · planner` when
  `model=stub`.
- **Shield interventions** — running count of moves where the blunder shield overrode Laya's raw
  pick.

## Desktop and WebGPU

Desktop browsers with WebGPU (Chrome, Edge) are recommended; the wasm fallback works but is
noticeably slower. The page needs a secure context (https, or localhost during development). The
`model=stub` scripted agent needs neither a model download nor WebGPU, and is the fastest way to
try the UI and the planner/shield logic on their own.

## Tests

- `web/apps/demo/test/chess/*` — unit tests for the planner (`candidates.test.ts`), the game
  wrapper (`game.test.ts`) and the policy/shield layer (`policy.test.ts`).
- `web/apps/demo/e2e/chess.spec.ts` — Playwright end-to-end tests: user moves and stub AI replies,
  illegal targets are ignored, switching sides, starting a new game, checkmate ending the game,
  a bad model URL surfacing an error, and (when `LAYA_MODEL_URL` is set) a real-model reply to
  1. e4 over WebGPU.
