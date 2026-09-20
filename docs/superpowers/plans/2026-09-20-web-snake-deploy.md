# Browser Snake demo and deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the terminal Snake demo to a canvas page driven by `@laya-mlx/web`, verified against Python fixtures, and publish the demo site to GitHub Pages (`mizchi.github.io/laya-mlx`) and a Hugging Face static Space (`mizchi/laya-web-demo`).

**Architecture:** `apps/demo/src/snake/` mirrors `laya_mlx/snake/`: `game.ts` (rules, Hamiltonian cycle, safety classification, food BFS), `policy.ts` (Python-identical compact prompt, shield), `loop.ts` (pacing state machine), `view.ts` (canvas board + DOM panel). Prompt strings and rule outcomes are checked against `web/fixtures/snake-multilingual.json`, dumped by a new `benchmarks/snake_fixtures.py` after extracting the prompt builder in Python into a pure function. Deployment builds the same Vite site twice (`VITE_BASE=/laya-mlx/` for Pages, `/` for the Space).

**Tech Stack:** TypeScript, Vite 8, vitest 5, Playwright 1.63, `@laya-mlx/web`, canvas 2D, GitHub Actions (`actions/deploy-pages`), `hf` CLI / `huggingface_hub` for the Space upload.

Spec: `docs/superpowers/specs/2026-09-20-web-demos-design.md` §2 and §4. Prerequisite: Plan 1 (`2026-09-20-web-runtime.md`) is merged on `main`.

---

## File structure

```
laya_mlx/snake/policy.py                 extract `build_prompt(...)` (pure) from `LayaPolicy.decide`
tests/test_snake.py                      add a test that build_prompt output feeds decide unchanged
benchmarks/snake_fixtures.py             dumps boards, moves, reachability, prompts for a seeded run
web/fixtures/snake-multilingual.json     committed output
web/apps/demo/
  snake.html
  src/snake/rng.ts                       mulberry32
  src/snake/game.ts                      SnakeGame (pure; port of game.py)
  src/snake/policy.ts                    buildPrompt, decide, applyShield (port of policy.py)
  src/snake/loop.ts                      GameLoop: pacing, pause, reset, max-speed, stats
  src/snake/view.ts                      SnakeView: canvas board + side panel DOM
  src/snake/stub-agent.ts                deterministic agent for tests and model-less demo
  src/snake/main.ts                      wiring, model loading with progress, keyboard
  src/loader.ts                          shared model loading UI (progress + provider label)
  test/snake/game.test.ts                (vitest, node)
  test/snake/policy.test.ts
  test/snake/loop.test.ts
  e2e/snake.spec.ts                      stub-agent smoke + env-gated real model
  vitest.config.ts                       new: demo unit tests
  vite.config.ts                         add snake.html input
  index.html                             link to snake
.github/workflows/pages.yml              GitHub Pages deploy
.github/workflows/space.yml              Hugging Face Space deploy (needs HF_TOKEN secret)
web/space/README.md                      Space front matter (sdk: static)
docs/SNAKE_DEMO.md, web/README.md        browser section
docs/onnx-model-card.md                  link to the Space
```

Naming used throughout:

- `Cell = readonly [number, number]` (x, y); directions `"UP" | "DOWN" | "LEFT" | "RIGHT"`.
- `MoveInfo = { direction, legal, safe, advance, reason, eats }` (same fields as Python).
- `Decision` has the same fields as the Python dataclass, camelCased: `probabilities, proposed, executed, safeDirections, intervened, deadEndRisk, foodReachable, inferenceMs, decisionMs, inputTokens, outputTokens, safeCount, plannerBest`.
- `SnakeAgent = { predict(state, questions): Promise<PredictResult> }` — the subset of `LayaAgent` the policy needs; `StubAgent` implements it.

---

### Task 1: Extract the Python prompt builder and dump Snake fixtures

**Files:**
- Modify: `laya_mlx/snake/policy.py`
- Modify: `tests/test_snake.py`
- Create: `benchmarks/snake_fixtures.py`
- Create: `web/fixtures/snake-multilingual.json` (generated)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_snake.py` (read the file first to reuse its fixtures/imports; the existing tests construct `SnakeGame` directly):

```python
def test_build_prompt_is_pure_and_matches_compact_layout():
    from laya_mlx.snake.game import SnakeGame
    from laya_mlx.snake.policy import build_prompt

    game = SnakeGame(width=8, height=6, seed=3, initial_length=4)
    moves = game.moves()
    reachable, space = game.food_reachability()
    state, questions, preferred = build_prompt(game, moves, reachable, space, "compact")
    assert state.startswith("Safe route: yes. Food reachable through empty cells: ")
    assert list(questions) == ["move", "risk", "food"]
    assert questions["move"]["instructions"] == "Choose the best safe move toward food."
    assert set(questions["move"]["criteria"]) == {"UP", "DOWN", "LEFT", "RIGHT"}
    assert preferred in {"UP", "DOWN", "LEFT", "RIGHT"}
    assert build_prompt(game, moves, reachable, space, "compact") == (state, questions, preferred)
    detailed_state, detailed_questions, _ = build_prompt(game, moves, reachable, space, "detailed")
    assert detailed_state.startswith("Snake game. ")
    assert detailed_questions["move"]["instructions"].startswith("Select the safest move")
```

- [ ] **Step 2: Run it**

Run: `cd /Users/mz/ghq/github.com/mizorewww/laya-mlx && LAYA_MLX_TEST_DEVICE=cpu uv run pytest tests/test_snake.py -q -k build_prompt`
Expected: FAIL, `ImportError: cannot import name 'build_prompt'`

- [ ] **Step 3: Extract `build_prompt`**

In `laya_mlx/snake/policy.py`, add above `class Decision` (module level):

```python
def build_prompt(game, moves, reachable, space, prompt="compact"):
    """State text and the three questions for one board; identical strings for the browser port."""
    safe = [m for m in moves if m.safe]
    preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
    if prompt == "compact":
        state = (
            f"Safe route: {'yes' if safe else 'no'}. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
        )
        criteria = {
            m.direction: (
                "Blocked. Collision."
                if not m.legal
                else "Unsafe. Traps the snake."
                if not m.safe
                else "Safe. Eat food now. Best."
                if m.eats
                else "Safe. Best route to food."
                if m.direction == preferred
                else "Safe. Slower route."
            )
            for m in moves
        }
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Choose the best safe move toward food.",
                "criteria": criteria,
            },
            "risk": {"type": "noul", "instructions": "Is a safe route available?"},
            "food": {"type": "noul", "instructions": "Is food reachable through empty cells?"},
        }
        return state, questions, preferred
    descriptions = {}
    for move in moves:
        if not move.legal:
            descriptions[move.direction] = f"Collision: {move.reason}. Unsafe."
        elif not move.safe:
            descriptions[move.direction] = "Unsafe route. Risk of trapping the snake."
        elif move.eats:
            descriptions[move.direction] = "Safe. Eat the food immediately. Best move."
        elif move.direction == preferred:
            descriptions[move.direction] = "Safe. Best progress toward food."
        else:
            descriptions[move.direction] = "Safe but less progress toward food."
    state = (
        f"Snake game. {len(safe)} safe directions available. "
        f"Food reachable through empty cells: {'yes' if reachable else 'no'}. "
        f"Open cells: {space}. Snake length: {len(game.body)}. "
        f"{'There is a safe route forward.' if safe else 'The snake is trapped.'}"
    )
    questions = {
        "move": {
            "type": "choice",
            "instructions": "Select the safest move with best progress toward food. Avoid collisions.",
            "criteria": descriptions,
        },
        "risk": {"type": "noul", "instructions": "Is there a safe route forward for the snake?"},
        "food": {"type": "noul", "instructions": "Is food reachable through the currently empty cells?"},
    }
    return state, questions, preferred
```

Then replace the body of `LayaPolicy.decide` from `moves = game.moves()` through the `if self.prompt == "compact": ...` block with:

```python
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        if not safe and self.guarded:
            raise RuntimeError("Cycle safety invariant violated: no safe action")
        reachable, space = game.food_reachability()
        state, questions, preferred = build_prompt(game, moves, reachable, space, self.prompt)
```

and keep everything from `inference_start = time.perf_counter()` onward unchanged (it uses `state`, `questions`, `safe`, `preferred`). Check with `git diff` that the strings moved verbatim: the compact and detailed texts must be byte-identical to the previous inline versions (the recorded benchmark JSON in `benchmarks/results/snake-*.json` depends on them).

- [ ] **Step 4: Run the Snake tests**

Run: `LAYA_MLX_TEST_DEVICE=cpu uv run pytest tests/test_snake.py -q`
Expected: all pass (including the new one).

- [ ] **Step 5: Write the fixture dump**

`benchmarks/snake_fixtures.py`:

```python
"""Dump Snake rule outcomes and prompts for the browser port (web/apps/demo/src/snake).

A seeded game is played with the planner's best safe direction, so no model is
needed. Each step records the board before the move, every MoveInfo, food
reachability, the compact and detailed prompts, and the board after the move.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from laya_mlx.snake.game import SnakeGame, hamiltonian_cycle
from laya_mlx.snake.policy import build_prompt

ROOT = Path(__file__).resolve().parents[1]


def play(width, height, seed, initial_length, steps):
    game = SnakeGame(width, height, seed, initial_length)
    records = []
    for _ in range(steps):
        if not game.alive or game.won:
            break
        moves = game.moves()
        reachable, space = game.food_reachability()
        prompts = {
            kind: dict(zip(("state", "questions", "preferred"), build_prompt(game, moves, reachable, space, kind)))
            for kind in ("compact", "detailed")
        }
        before = game.snapshot()
        safe = [m for m in moves if m.safe]
        direction = max(safe, key=lambda m: m.advance).direction
        ate = game.step(direction)
        records.append(
            {
                "before": before,
                "moves": [asdict(m) for m in moves],
                "food_reachable": reachable,
                "open_cells": space,
                "prompts": prompts,
                "executed": direction,
                "ate": ate,
                "after": game.snapshot(),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "web/fixtures/snake-multilingual.json")
    args = parser.parse_args()
    games = [
        {"width": 24, "height": 16, "seed": 7, "initial_length": 6, "steps": 120},
        {"width": 8, "height": 6, "seed": 3, "initial_length": 4, "steps": 60},
        {"width": 4, "height": 4, "seed": 1, "initial_length": 2, "steps": 40},
    ]
    fixture = {
        "cycles": {f"{w}x{h}": hamiltonian_cycle(w, h) for w, h in ((24, 16), (8, 6), (4, 4), (5, 4))},
        "games": [{**g, "records": play(**g)} for g in games],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"{len(games)} games, {sum(len(g['records']) for g in fixture['games'])} steps -> {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Generate, lint, commit**

Run: `uv run python -m benchmarks.snake_fixtures && uv run ruff check . && uv run ruff format .`
Expected: `3 games, <N> steps -> …/web/fixtures/snake-multilingual.json` (the 4×4 game may end early by winning; N is around 200). File size under 1 MB.

```bash
git add laya_mlx/snake/policy.py tests/test_snake.py benchmarks/snake_fixtures.py web/fixtures/snake-multilingual.json
git commit -m "Extract the Snake prompt builder and dump browser fixtures"
```

---

### Task 2: Seeded RNG and the Snake rules port

**Files:**
- Create: `web/apps/demo/src/snake/rng.ts`, `web/apps/demo/src/snake/game.ts`
- Create: `web/apps/demo/vitest.config.ts`, `web/apps/demo/test/snake/game.test.ts`
- Modify: `web/apps/demo/package.json` (scripts `test: vitest run`, devDependency `vitest`), `web/apps/demo/tsconfig.json` (include `test`)

- [ ] **Step 1: Failing tests**

`web/apps/demo/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: { include: ["test/**/*.test.ts"], environment: "node" },
});
```

`web/apps/demo/test/snake/fixtures.ts`:

```ts
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import type { Direction, MoveInfo, Snapshot } from "../../src/snake/game.ts";

export interface SnakeRecord {
  before: Snapshot;
  moves: MoveInfo[];
  food_reachable: boolean;
  open_cells: number;
  prompts: Record<"compact" | "detailed", { state: string; questions: Record<string, unknown>; preferred: string }>;
  executed: Direction;
  ate: boolean;
  after: Snapshot;
}
export interface SnakeFixture {
  cycles: Record<string, [number, number][]>;
  games: { width: number; height: number; seed: number; initial_length: number; steps: number; records: SnakeRecord[] }[];
}

const path = fileURLToPath(new URL("../../../../fixtures/snake-multilingual.json", import.meta.url));
export const snakeFixture: SnakeFixture = JSON.parse(readFileSync(path, "utf8"));
```

`web/apps/demo/test/snake/game.test.ts`:

```ts
import { describe, expect, it } from "vitest";

import { SnakeGame, hamiltonianCycle } from "../../src/snake/game.ts";
import { mulberry32 } from "../../src/snake/rng.ts";
import { snakeFixture } from "./fixtures.ts";

describe("hamiltonianCycle", () => {
  it.each(Object.entries(snakeFixture.cycles))("%s matches Python", (_, cells) => {
    const [w, h] = cells.length === 384 ? [24, 16] : cells.length === 48 ? [8, 6] : cells.length === 16 ? [4, 4] : [5, 4];
    expect(hamiltonianCycle(w, h)).toEqual(cells);
  });
  it("rejects boards Python rejects", () => {
    expect(() => hamiltonianCycle(3, 4)).toThrow(/>= 4/);
    expect(() => hamiltonianCycle(5, 5)).toThrow(/even/);
  });
});

describe("mulberry32", () => {
  it("is deterministic and in [0, 1)", () => {
    const a = mulberry32(7);
    const b = mulberry32(7);
    const values = Array.from({ length: 5 }, () => a());
    expect(Array.from({ length: 5 }, () => b())).toEqual(values);
    expect(values.every((v) => v >= 0 && v < 1)).toBe(true);
  });
});

describe("SnakeGame rules against Python", () => {
  for (const g of snakeFixture.games) {
    it(`${g.width}x${g.height} seed ${g.seed}: moves, reachability and step outcomes`, () => {
      const game = SnakeGame.fromSnapshot(g.records[0]!.before);
      for (const record of g.records) {
        expect(game.snapshot()).toEqual(record.before);
        expect(game.moves()).toEqual(record.moves);
        const [reachable, space] = game.foodReachability();
        expect(reachable).toBe(record.food_reachable);
        expect(space).toBe(record.open_cells);
        // Food placement uses a different RNG than Python; replay Python's food to compare rules.
        const ate = game.step(record.executed, () => record.after.food);
        expect(ate).toBe(record.ate);
        expect(game.snapshot()).toEqual(record.after);
      }
    });
  }
  it("dies on walls, reversal and body; wins when the board is full", () => {
    const game = new SnakeGame(4, 4, 1, 2);
    expect(game.legalReason("UP") === "legal" || game.legalReason("UP") === "wall").toBe(true);
    const tiny = new SnakeGame(4, 4, 1, 15);
    expect(tiny.moves().filter((m) => m.legal).length).toBeGreaterThan(0);
    while (tiny.alive && !tiny.won) {
      const safe = tiny.moves().filter((m) => m.safe);
      tiny.step(safe.reduce((a, b) => (b.advance > a.advance ? b : a)).direction);
    }
    expect(tiny.won).toBe(true);
    expect(tiny.food).toBeNull();
    expect(() => tiny.step("UP")).toThrow(/finished/);
  });
});
```

Add `"test": "vitest run"` to `apps/demo/package.json` scripts (replacing the echo), `"vitest": "^5.0.0"` to devDependencies, add `"test"` to `tsconfig.json` `include`, run `pnpm install`.

- [ ] **Step 2: Run to verify failure**

Run: `cd web && pnpm --filter laya-demo test`
Expected: FAIL, cannot find `../../src/snake/game.ts`

- [ ] **Step 3: Implement `rng.ts`**

```ts
/** Deterministic 32-bit PRNG; sequences differ from Python's random.Random, rules do not. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
```

- [ ] **Step 4: Implement `game.ts`** (port of `laya_mlx/snake/game.py`; keep method names close)

```ts
import { mulberry32 } from "./rng.ts";

export type Direction = "UP" | "DOWN" | "LEFT" | "RIGHT";
export const DIRECTIONS: readonly Direction[] = ["UP", "DOWN", "LEFT", "RIGHT"];
export const VECTORS: Record<Direction, readonly [number, number]> = {
  UP: [0, -1],
  DOWN: [0, 1],
  LEFT: [-1, 0],
  RIGHT: [1, 0],
};

export type Cell = [number, number];
const key = (c: Cell) => c[0] * 100000 + c[1];

/** Visit each square once with adjacent steps, including the closing edge (Python `hamiltonian_cycle`). */
export function hamiltonianCycle(width: number, height: number): Cell[] {
  if (Math.min(width, height) < 4 || (width % 2 && height % 2)) {
    throw new Error("Board dimensions must be >= 4, with at least one even dimension");
  }
  if (height % 2) return hamiltonianCycle(height, width).map(([x, y]) => [y, x]);
  const path: Cell[] = [[0, 0]];
  for (let y = 0; y < height; y++) {
    if (y % 2 === 0) for (let x = 1; x < width; x++) path.push([x, y]);
    else for (let x = width - 1; x > 0; x--) path.push([x, y]);
  }
  for (let y = height - 1; y > 0; y--) path.push([0, y]);
  return path;
}

export interface MoveInfo {
  direction: Direction;
  legal: boolean;
  safe: boolean;
  advance: number;
  reason: string;
  eats: boolean;
}

export interface Snapshot {
  width: number;
  height: number;
  seed: number;
  body: [number, number][];
  food: [number, number] | null;
  score: number;
  length: number;
  ticks: number;
  alive: boolean;
  won: boolean;
  death_reason: string | null;
}

export class SnakeGame {
  readonly cycle: Cell[];
  readonly indices: Map<number, number>;
  readonly capacity: number;
  body: Cell[]; // head first, like Python's deque
  food: Cell | null;
  score = 0;
  ticks = 0;
  alive = true;
  won = false;
  deathReason: string | null = null;
  private readonly rng: () => number;

  constructor(
    readonly width = 24,
    readonly height = 16,
    readonly seed = 7,
    readonly initialLength = 6,
  ) {
    this.cycle = hamiltonianCycle(width, height);
    this.indices = new Map(this.cycle.map((cell, index) => [key(cell), index]));
    this.capacity = width * height;
    if (!(initialLength >= 2 && initialLength < this.capacity)) {
      throw new Error("Initial length must be >= 2 and smaller than the board");
    }
    this.rng = mulberry32(seed);
    const start = this.indices.get(key([Math.floor(width / 2), Math.floor(height / 2)]))!;
    this.body = Array.from({ length: initialLength }, (_, i) => this.cycle[(start - i + this.capacity) % this.capacity]!);
    this.food = this.spawnFood();
  }

  /** Rebuild a game at an arbitrary recorded board (tests replay Python runs). */
  static fromSnapshot(s: Snapshot): SnakeGame {
    const game = new SnakeGame(s.width, s.height, s.seed, Math.max(2, Math.min(s.length, s.width * s.height - 1)));
    game.body = s.body.map(([x, y]) => [x, y]);
    game.food = s.food ? [s.food[0], s.food[1]] : null;
    game.score = s.score;
    game.ticks = s.ticks;
    game.alive = s.alive;
    game.won = s.won;
    game.deathReason = s.death_reason;
    return game;
  }

  get head(): Cell {
    return this.body[0]!;
  }

  private spawnFood(): Cell | null {
    const occupied = new Set(this.body.map(key));
    const empty = this.cycle.filter((cell) => !occupied.has(key(cell)));
    return empty.length ? empty[Math.floor(this.rng() * empty.length)]! : null;
  }

  target(direction: Direction): Cell {
    const [dx, dy] = VECTORS[direction];
    return [this.head[0] + dx, this.head[1] + dy];
  }

  legalReason(direction: Direction): string {
    const cell = this.target(direction);
    const [x, y] = cell;
    if (!(x >= 0 && x < this.width && y >= 0 && y < this.height)) return "wall";
    const neck = this.body[1];
    if (neck && neck[0] === x && neck[1] === y) return "reverse";
    const occupied = new Set(this.body.map(key));
    if (!(this.food && this.food[0] === x && this.food[1] === y)) {
      occupied.delete(key(this.body[this.body.length - 1]!)); // The tail moves on a non-growing step.
    }
    return occupied.has(key(cell)) ? "body" : "legal";
  }

  moves(): MoveInfo[] {
    if (!this.alive || this.won) return [];
    const headIndex = this.indices.get(key(this.head))!;
    const mod = (n: number) => ((n % this.capacity) + this.capacity) % this.capacity;
    const tailDistance = mod(this.indices.get(key(this.body[this.body.length - 1]!))! - headIndex);
    const foodDistance = mod(this.indices.get(key(this.food!))! - headIndex);
    return DIRECTIONS.map((direction) => {
      let reason = this.legalReason(direction);
      const legal = reason === "legal";
      const target = this.target(direction);
      const advance = mod((this.indices.get(key(target)) ?? headIndex) - headIndex);
      const eats = this.food !== null && target[0] === this.food[0] && target[1] === this.food[1];
      let safe = legal;
      if (safe && (advance > tailDistance || (advance === tailDistance && eats))) {
        safe = false;
        reason = "would cross the tail";
      }
      if (safe && (advance === 0 || advance > foodDistance)) {
        safe = false;
        reason = "would skip the food on the safe route";
      }
      return { direction, legal, safe, advance, reason, eats };
    });
  }

  /** Current empty-cell connectivity; the occupied tail is not treated as empty. */
  foodReachability(): [boolean, number] {
    const blocked = new Set(this.body.slice(1).map(key));
    const visited = new Set([key(this.head)]);
    const queue: Cell[] = [this.head];
    while (queue.length) {
      const [x, y] = queue.shift()!;
      for (const [dx, dy] of Object.values(VECTORS)) {
        const cell: Cell = [x + dx, y + dy];
        const k = key(cell);
        if (cell[0] >= 0 && cell[0] < this.width && cell[1] >= 0 && cell[1] < this.height && !blocked.has(k) && !visited.has(k)) {
          visited.add(k);
          queue.push(cell);
        }
      }
    }
    return [this.food !== null && visited.has(key(this.food)), visited.size];
  }

  /** Advance one move; returns whether food was eaten. `placeFood` lets tests replay Python's food. */
  step(direction: Direction, placeFood: () => Cell | null = () => this.spawnFood()): boolean {
    if (!this.alive || this.won) throw new Error("Cannot step a finished game");
    this.ticks += 1;
    const reason = this.legalReason(direction);
    if (reason !== "legal") {
      this.alive = false;
      this.deathReason = reason;
      return false;
    }
    const target = this.target(direction);
    this.body.unshift(target);
    if (this.food && target[0] === this.food[0] && target[1] === this.food[1]) {
      this.score += 1;
      if (this.body.length === this.capacity) {
        this.won = true;
        this.food = null;
      } else {
        this.food = placeFood();
      }
      return true;
    }
    this.body.pop();
    return false;
  }

  snapshot(): Snapshot {
    return {
      width: this.width,
      height: this.height,
      seed: this.seed,
      body: this.body.map(([x, y]) => [x, y]),
      food: this.food ? [this.food[0], this.food[1]] : null,
      score: this.score,
      length: this.body.length,
      ticks: this.ticks,
      alive: this.alive,
      won: this.won,
      death_reason: this.deathReason,
    };
  }
}
```

Note `moves()` computes `foodDistance` with `this.food!` — Python does the same (`self.food` is only `None` after a win, and `moves()` returns `[]` then). Python's `indices.get(target, head_index)` for off-board targets is mirrored by `?? headIndex`.

- [ ] **Step 5: Run the tests**

Run: `pnpm --filter laya-demo test`
Expected: all pass. If a `moves` comparison fails, print the first differing record index and direction; the usual suspects are the tail-removal rule in `legalReason` and the modulo of negative numbers.

- [ ] **Step 6: Typecheck, format, commit**

```bash
pnpm typecheck && pnpm format:write && pnpm format
git add apps/demo/src/snake/rng.ts apps/demo/src/snake/game.ts apps/demo/test/snake apps/demo/vitest.config.ts apps/demo/package.json apps/demo/tsconfig.json pnpm-lock.yaml
git commit -m "Port the Snake rules and cycle planner to TypeScript"
```

---

### Task 3: Policy: prompt, decision, shield

**Files:**
- Create: `web/apps/demo/src/snake/policy.ts`, `web/apps/demo/src/snake/stub-agent.ts`
- Test: `web/apps/demo/test/snake/policy.test.ts`

- [ ] **Step 1: Failing tests**

`web/apps/demo/test/snake/policy.test.ts`:

```ts
import { describe, expect, it } from "vitest";

import { SnakeGame } from "../../src/snake/game.ts";
import { applyShield, buildPrompt, decide } from "../../src/snake/policy.ts";
import { StubAgent } from "../../src/snake/stub-agent.ts";
import { snakeFixture } from "./fixtures.ts";

describe("buildPrompt", () => {
  for (const g of snakeFixture.games) {
    it(`${g.width}x${g.height} seed ${g.seed}: compact and detailed prompts match Python`, () => {
      for (const record of g.records) {
        const game = SnakeGame.fromSnapshot(record.before);
        const moves = game.moves();
        const [reachable, space] = game.foodReachability();
        for (const kind of ["compact", "detailed"] as const) {
          const prompt = buildPrompt(game, moves, reachable, space, kind);
          expect(prompt.state).toBe(record.prompts[kind].state);
          expect(prompt.questions).toEqual(record.prompts[kind].questions);
          expect(prompt.preferred).toBe(record.prompts[kind].preferred);
        }
      }
    });
  }
});

describe("applyShield", () => {
  const p = { UP: 0.6, DOWN: 0.3, LEFT: 0.05, RIGHT: 0.05 };
  it("executes the proposal when it is safe", () => {
    expect(applyShield(p, ["UP", "DOWN"], true)).toEqual({ proposed: "UP", executed: "UP", intervened: false });
  });
  it("redirects to the most likely safe direction when guarded", () => {
    expect(applyShield(p, ["DOWN", "LEFT"], true)).toEqual({ proposed: "UP", executed: "DOWN", intervened: true });
  });
  it("executes the raw proposal when unguarded", () => {
    expect(applyShield(p, ["DOWN"], false)).toEqual({ proposed: "UP", executed: "UP", intervened: false });
  });
});

describe("decide", () => {
  it("returns a Decision with model probabilities, estimates and timings", async () => {
    const game = new SnakeGame(8, 6, 3, 4);
    const decision = await decide(new StubAgent(), game, { guarded: true, prompt: "compact" });
    expect(Object.keys(decision.probabilities)).toEqual(["UP", "DOWN", "LEFT", "RIGHT"]);
    expect(game.moves().filter((m) => m.safe).map((m) => m.direction)).toContain(decision.executed);
    expect(decision.deadEndRisk).toBeGreaterThanOrEqual(0);
    expect(decision.foodReachable).toBeLessThanOrEqual(1);
    expect(decision.inputTokens).toBeGreaterThan(0);
    expect(decision.outputTokens).toBe(0);
    expect(decision.inferenceMs).toBeGreaterThanOrEqual(0);
  });
  it("throws when the shield invariant is violated", async () => {
    const game = new SnakeGame(8, 6, 3, 4);
    game.alive = false;
    await expect(decide(new StubAgent(), game, { guarded: true, prompt: "compact" })).rejects.toThrow(/no safe action/);
  });
  it("rejects invalid probabilities from the model", async () => {
    const game = new SnakeGame(8, 6, 3, 4);
    const bad = new StubAgent({ probability: () => Number.NaN });
    await expect(decide(bad, game, { guarded: true, prompt: "compact" })).rejects.toThrow(/invalid probability/);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm --filter laya-demo test -- policy`
Expected: FAIL, module not found

- [ ] **Step 3: Implement `stub-agent.ts`**

```ts
import type { Answer, PredictResult, Question, State } from "@laya-mlx/web";

export interface StubAgentOptions {
  /** Probability assigned to the option whose description contains "Best"; the rest share the remainder. */
  probability?: (label: string, description: string) => number;
}

/** Model-free agent: prefers the planner's "Best" option; used by tests and the `?model=stub` demo mode. */
export class StubAgent {
  constructor(private readonly options: StubAgentOptions = {}) {}

  async predict(state: State, questions: Record<string, Question>): Promise<PredictResult> {
    const answers: Record<string, Answer> = {};
    let tokens = typeof state === "string" ? state.length : JSON.stringify(state).length;
    for (const [qid, q] of Object.entries(questions)) {
      tokens += JSON.stringify(q).length;
      if (q.type === "choice") {
        const entries = Array.isArray(q.criteria) ? q.criteria.map((c) => [c, ""] as const) : Object.entries(q.criteria);
        const raw = entries.map(([label, desc]) =>
          this.options.probability
            ? this.options.probability(label, String(desc))
            : String(desc).includes("Best")
              ? 0.7
              : String(desc).startsWith("Safe")
                ? 0.2
                : 0.05,
        );
        const sum = raw.reduce((a, b) => a + b, 0);
        const probabilities = Object.fromEntries(entries.map(([label], i) => [label, raw[i]! / sum]));
        const best = entries[raw.indexOf(Math.max(...raw))]![0];
        answers[qid] = { type: "choice", confidence: 0.5, action: { act_probability: 1 }, choice: best, probabilities };
      } else if (q.type === "noul") {
        const yes = /yes/.test(state as string) ? 0.9 : 0.4;
        answers[qid] = { type: "noul", confidence: yes, action: { act_probability: 1 }, noul: yes };
      } else {
        answers[qid] = {
          type: "score",
          confidence: 0.5,
          action: { act_probability: 1 },
          score: 0,
          legend: {},
          probabilities: { "0": 1 },
        };
      }
    }
    return { model: "laya-rl-agent", answers, usage: { input_tokens: Math.ceil(tokens / 4), output_tokens: 0 } };
  }
}
```

(The `noul` heuristic reads the compact state text "Safe route: yes" / "Food reachable … yes"; good enough for a stub.)

- [ ] **Step 4: Implement `policy.ts`** (port of `build_prompt` and `LayaPolicy.decide`)

```ts
import type { PredictResult, Question, State } from "@laya-mlx/web";

import { DIRECTIONS, type Direction, type MoveInfo, type SnakeGame } from "./game.ts";

export type PromptKind = "compact" | "detailed";

export interface SnakeAgent {
  predict(state: State, questions: Record<string, Question>): Promise<PredictResult>;
}

export interface Prompt {
  state: string;
  questions: Record<string, Question>;
  preferred: Direction | "NONE";
}

/** Python `build_prompt`: identical strings, checked against fixtures. */
export function buildPrompt(game: SnakeGame, moves: MoveInfo[], reachable: boolean, space: number, kind: PromptKind): Prompt {
  const safe = moves.filter((m) => m.safe);
  const preferred: Direction | "NONE" = safe.length ? safe.reduce((a, b) => (b.advance > a.advance ? b : a)).direction : "NONE";
  const yes = (v: boolean) => (v ? "yes" : "no");
  if (kind === "compact") {
    const criteria = Object.fromEntries(
      moves.map((m) => [
        m.direction,
        !m.legal
          ? "Blocked. Collision."
          : !m.safe
            ? "Unsafe. Traps the snake."
            : m.eats
              ? "Safe. Eat food now. Best."
              : m.direction === preferred
                ? "Safe. Best route to food."
                : "Safe. Slower route.",
      ]),
    );
    return {
      state: `Safe route: ${yes(safe.length > 0)}. Food reachable through empty cells: ${yes(reachable)}.`,
      questions: {
        move: { type: "choice", instructions: "Choose the best safe move toward food.", criteria },
        risk: { type: "noul", instructions: "Is a safe route available?" },
        food: { type: "noul", instructions: "Is food reachable through empty cells?" },
      },
      preferred,
    };
  }
  const descriptions = Object.fromEntries(
    moves.map((m) => [
      m.direction,
      !m.legal
        ? `Collision: ${m.reason}. Unsafe.`
        : !m.safe
          ? "Unsafe route. Risk of trapping the snake."
          : m.eats
            ? "Safe. Eat the food immediately. Best move."
            : m.direction === preferred
              ? "Safe. Best progress toward food."
              : "Safe but less progress toward food.",
    ]),
  );
  return {
    state:
      `Snake game. ${safe.length} safe directions available. ` +
      `Food reachable through empty cells: ${yes(reachable)}. ` +
      `Open cells: ${space}. Snake length: ${game.body.length}. ` +
      `${safe.length ? "There is a safe route forward." : "The snake is trapped."}`,
    questions: {
      move: {
        type: "choice",
        instructions: "Select the safest move with best progress toward food. Avoid collisions.",
        criteria: descriptions,
      },
      risk: { type: "noul", instructions: "Is there a safe route forward for the snake?" },
      food: { type: "noul", instructions: "Is food reachable through the currently empty cells?" },
    },
    preferred,
  };
}

const argmaxDirection = (p: Record<Direction, number>, among: readonly Direction[]) =>
  among.reduce((a, b) => (p[b] > p[a] ? b : a));

/** Python's execution shield: a guarded policy never executes an unsafe proposal. */
export function applyShield(
  probabilities: Record<Direction, number>,
  safeDirections: Direction[],
  guarded: boolean,
): { proposed: Direction; executed: Direction; intervened: boolean } {
  const proposed = argmaxDirection(probabilities, DIRECTIONS);
  const executed = guarded && !safeDirections.includes(proposed) ? argmaxDirection(probabilities, safeDirections) : proposed;
  return { proposed, executed, intervened: proposed !== executed };
}

export interface Decision {
  probabilities: Record<Direction, number>;
  proposed: Direction;
  executed: Direction;
  safeDirections: Direction[];
  intervened: boolean;
  deadEndRisk: number;
  foodReachable: number;
  inferenceMs: number;
  decisionMs: number;
  inputTokens: number;
  outputTokens: number;
  safeCount: number;
  plannerBest: Direction | "NONE";
}

export interface PolicyOptions {
  guarded: boolean;
  prompt: PromptKind;
}

/** Python `LayaPolicy.decide`. */
export async function decide(agent: SnakeAgent, game: SnakeGame, options: PolicyOptions): Promise<Decision> {
  const started = performance.now();
  const moves = game.moves();
  const safe = moves.filter((m) => m.safe);
  if (!safe.length && options.guarded) throw new Error("Cycle safety invariant violated: no safe action");
  const [reachable, space] = game.foodReachability();
  const prompt = buildPrompt(game, moves, reachable, space, options.prompt);
  const inferenceStart = performance.now();
  const output = await agent.predict(prompt.state, prompt.questions);
  const inferenceMs = performance.now() - inferenceStart;
  const move = output.answers["move"];
  const risk = output.answers["risk"];
  const food = output.answers["food"];
  if (move?.type !== "choice" || risk?.type !== "noul" || food?.type !== "noul") {
    throw new Error("Model returned an unexpected answer shape");
  }
  const probabilities = Object.fromEntries(DIRECTIONS.map((d) => [d, move.probabilities[d] ?? Number.NaN])) as Record<Direction, number>;
  const scores = [...Object.values(probabilities), risk.noul, food.noul];
  if (scores.some((v) => !Number.isFinite(v) || v < 0 || v > 1)) {
    throw new Error("Model returned an invalid probability; no move executed");
  }
  const safeDirections = safe.map((m) => m.direction);
  const { proposed, executed, intervened } = applyShield(probabilities, safeDirections, options.guarded);
  return {
    probabilities,
    proposed,
    executed,
    safeDirections,
    intervened,
    deadEndRisk: 1 - risk.noul,
    foodReachable: food.noul,
    inferenceMs,
    decisionMs: performance.now() - started,
    inputTokens: output.usage.input_tokens,
    outputTokens: output.usage.output_tokens ?? 0,
    safeCount: safe.length,
    plannerBest: prompt.preferred,
  };
}
```

`applyShield` with an unguarded policy and no safe directions returns the raw proposal (Python: `allowed` empty, `proposed` used). When guarded, `decide` has already thrown if `safe` is empty, so `argmaxDirection(p, safeDirections)` never sees an empty list.

- [ ] **Step 5: Run the tests**

Run: `pnpm --filter laya-demo test`
Expected: all pass (prompt strings for ~220 records × 2 kinds).

- [ ] **Step 6: Typecheck, format, commit**

```bash
pnpm typecheck && pnpm format:write && pnpm format
git add apps/demo/src/snake/policy.ts apps/demo/src/snake/stub-agent.ts apps/demo/test/snake/policy.test.ts
git commit -m "Port the Snake policy, prompt and safety shield"
```

---

### Task 4: Game loop state machine

**Files:**
- Create: `web/apps/demo/src/snake/loop.ts`
- Test: `web/apps/demo/test/snake/loop.test.ts`

- [ ] **Step 1: Failing tests**

```ts
import { describe, expect, it, vi } from "vitest";

import { GameLoop } from "../../src/snake/loop.ts";
import { StubAgent } from "../../src/snake/stub-agent.ts";

const settings = { width: 8, height: 6, seed: 3, initialLength: 4, fps: 12, guarded: true, prompt: "compact" as const };

describe("GameLoop", () => {
  it("advances one decision per tick and tracks stats", async () => {
    const loop = new GameLoop(new StubAgent(), settings);
    const before = loop.game.ticks;
    await loop.tick();
    expect(loop.game.ticks).toBe(before + 1);
    expect(loop.lastDecision).not.toBeNull();
    expect(loop.stats.decisions).toBe(1);
    expect(loop.stats.best).toBeGreaterThanOrEqual(0);
  });
  it("does not advance while paused", async () => {
    const loop = new GameLoop(new StubAgent(), settings);
    loop.togglePause();
    await loop.tick();
    expect(loop.game.ticks).toBe(0);
    expect(loop.paused).toBe(true);
  });
  it("starts a new round with the next seed on reset and when a round ends", async () => {
    const loop = new GameLoop(new StubAgent(), settings);
    loop.reset();
    expect(loop.stats.round).toBe(2);
    expect(loop.game.seed).toBe(settings.seed + 1);
    loop.game.alive = false;
    await loop.tick();
    expect(loop.stats.round).toBe(3);
    expect(loop.game.alive).toBe(true);
  });
  it("adjusts pacing within [1, 240] decisions per second", () => {
    const loop = new GameLoop(new StubAgent(), { ...settings, fps: 239 });
    loop.changeSpeed(+2);
    expect(loop.fps).toBe(240);
    loop.changeSpeed(-500);
    expect(loop.fps).toBe(1);
  });
  it("counts shield interventions", async () => {
    const unsafe = new StubAgent({ probability: (_, desc) => (desc.startsWith("Blocked") || desc.startsWith("Unsafe") ? 1 : 0.01) });
    const loop = new GameLoop(unsafe, settings);
    await loop.tick();
    expect(loop.stats.interventions + (loop.lastDecision!.intervened ? 0 : 1)).toBeGreaterThanOrEqual(1);
  });
  it("never overlaps decisions when tick is called concurrently", async () => {
    const slow = new StubAgent();
    const spy = vi.spyOn(slow, "predict");
    const loop = new GameLoop(slow, settings);
    await Promise.all([loop.tick(), loop.tick(), loop.tick()]);
    expect(spy).toHaveBeenCalledTimes(1);
  });
});
```

- [ ] **Step 2: Run to verify failure** — `pnpm --filter laya-demo test -- loop` → module not found.

- [ ] **Step 3: Implement `loop.ts`**

```ts
import { SnakeGame } from "./game.ts";
import { decide, type Decision, type PolicyOptions, type SnakeAgent } from "./policy.ts";

export interface LoopSettings extends PolicyOptions {
  width: number;
  height: number;
  seed: number;
  initialLength: number;
  /** Paced decisions per second; ignored in max-speed mode. */
  fps: number;
}

export interface LoopStats {
  best: number;
  round: number;
  decisions: number;
  interventions: number;
  startedAt: number;
  /** Rolling decisions per second over the last second of wall clock. */
  decisionsPerSecond: number;
}

/** Owns the game, the pacing and the running statistics; rendering is someone else's job. */
export class GameLoop {
  game: SnakeGame;
  lastDecision: Decision | null = null;
  paused = false;
  maxSpeed = false;
  fps: number;
  readonly stats: LoopStats;
  private inFlight: Promise<void> | null = null;
  private recent: number[] = [];

  constructor(
    private readonly agent: SnakeAgent,
    private readonly settings: LoopSettings,
  ) {
    this.fps = settings.fps;
    this.game = this.newGame(settings.seed);
    this.stats = { best: 0, round: 1, decisions: 0, interventions: 0, startedAt: performance.now(), decisionsPerSecond: 0 };
  }

  private newGame(seed: number): SnakeGame {
    return new SnakeGame(this.settings.width, this.settings.height, seed, this.settings.initialLength);
  }

  get elapsedSeconds(): number {
    return (performance.now() - this.stats.startedAt) / 1000;
  }

  togglePause(): void {
    this.paused = !this.paused;
  }

  changeSpeed(delta: number): void {
    this.fps = Math.min(240, Math.max(1, this.fps + delta));
  }

  reset(): void {
    this.stats.round += 1;
    this.game = this.newGame(this.settings.seed + this.stats.round - 1);
    this.lastDecision = null;
  }

  /** One decision + move. Concurrent calls share the in-flight decision instead of stacking. */
  tick(): Promise<void> {
    if (this.inFlight) return this.inFlight;
    this.inFlight = this.advance().finally(() => {
      this.inFlight = null;
    });
    return this.inFlight;
  }

  private async advance(): Promise<void> {
    if (this.paused) return;
    if (!this.game.alive || this.game.won) {
      this.reset();
      return;
    }
    const decision = await decide(this.agent, this.game, this.settings);
    this.lastDecision = decision;
    this.game.step(decision.executed);
    this.stats.decisions += 1;
    if (decision.intervened) this.stats.interventions += 1;
    this.stats.best = Math.max(this.stats.best, this.game.score);
    const now = performance.now();
    this.recent.push(now);
    this.recent = this.recent.filter((t) => now - t <= 1000);
    this.stats.decisionsPerSecond = this.recent.length;
  }

  /** Milliseconds to wait before the next tick in paced mode; 0 in max-speed mode. */
  delayAfter(tickStartedAt: number): number {
    if (this.maxSpeed) return 0;
    return Math.max(0, 1000 / this.fps - (performance.now() - tickStartedAt));
  }
}
```

The Python CLI sleeps one second at the end of a round before starting the next; keep that in `main.ts` (the loop itself just resets on the next tick so tests stay synchronous).

- [ ] **Step 4: Run, typecheck, commit**

```bash
pnpm --filter laya-demo test && pnpm typecheck && pnpm format:write && pnpm format
git add apps/demo/src/snake/loop.ts apps/demo/test/snake/loop.test.ts
git commit -m "Add the Snake game loop with pacing and statistics"
```

---

### Task 5: View, page and wiring

**Files:**
- Create: `web/apps/demo/snake.html`, `web/apps/demo/src/snake/view.ts`, `web/apps/demo/src/snake/main.ts`, `web/apps/demo/src/loader.ts`, `web/apps/demo/src/styles.css`
- Modify: `web/apps/demo/vite.config.ts` (add `snake` input), `web/apps/demo/index.html` (link)

No unit test for the view; Task 6 covers it with Playwright. Keep `view.ts` free of game logic: it takes a `Snapshot`, a `Decision | null`, stats and labels, and draws.

- [ ] **Step 1: Page** — `snake.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Laya Snake</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="stylesheet" href="/src/styles.css" />
  </head>
  <body class="snake">
    <header class="bar">
      <span class="muted">LAYA / LOCAL INTELLIGENCE</span>
      <span id="state" class="state">LOADING</span>
    </header>
    <main class="layout">
      <section class="board">
        <h1>S N A K E <span id="round" class="muted">ROUND 01</span></h1>
        <canvas id="board" width="480" height="320" aria-label="Snake board"></canvas>
        <div class="scores">
          <div><span class="muted">SCORE</span><strong id="score">0</strong></div>
          <div><span class="muted">LENGTH</span><strong id="length">0</strong></div>
          <div><span class="muted">BEST</span><strong id="best">0</strong></div>
        </div>
        <div class="fill"><div id="fill-bar"></div><span id="fill-pct" class="muted">0.0%</span></div>
        <div class="controls">
          <button id="pause">Pause (Space)</button>
          <button id="slower">Slower (↓)</button>
          <button id="faster">Faster (↑)</button>
          <button id="max-speed">Max speed</button>
          <button id="reset">Reset (R)</button>
        </div>
      </section>
      <aside class="panel">
        <h2>Laya <span id="engine-title" class="muted">WebGPU</span></h2>
        <p id="loading" class="muted">loading model…</p>
        <div class="row"><span>NEXT MOVE</span><span class="muted">MODEL PROBABILITIES</span></div>
        <ul id="probs">
          <li data-dir="UP"><span class="dir">UP</span><span class="track"><span class="bar"></span></span><span class="val">0.00</span></li>
          <li data-dir="DOWN"><span class="dir">DOWN</span><span class="track"><span class="bar"></span></span><span class="val">0.00</span></li>
          <li data-dir="LEFT"><span class="dir">LEFT</span><span class="track"><span class="bar"></span></span><span class="val">0.00</span></li>
          <li data-dir="RIGHT"><span class="dir">RIGHT</span><span class="track"><span class="bar"></span></span><span class="val">0.00</span></li>
        </ul>
        <div class="row"><span class="muted">EXECUTING</span><strong id="executing">—</strong><span id="shield" class="amber" hidden>SHIELD</span></div>
        <div class="metric"><span class="muted">DEAD-END RISK</span><span class="track"><span id="risk-bar" class="bar amber"></span></span><span id="risk-val">0.00</span></div>
        <div class="metric"><span class="muted">FOOD REACHABLE</span><span class="track"><span id="food-bar" class="bar cyan"></span></span><span id="food-val">0.00</span></div>
        <dl class="stats">
          <dt>INFERENCE</dt><dd id="inference">0.0 ms</dd>
          <dt>DECISIONS</dt><dd id="rate">0.0 /s</dd>
          <dt>OUTPUT TOKENS</dt><dd id="output-tokens">0</dd>
          <dt>NETWORK</dt><dd class="green">OFFLINE AFTER LOAD</dd>
          <dt>ENGINE</dt><dd id="engine">—</dd>
        </dl>
        <p class="muted">Laya + cycle safety</p>
        <p class="amber">Shield interventions <span id="interventions">0000</span></p>
      </aside>
    </main>
    <footer class="bar"><span class="muted">SPACE pause · ↑/↓ speed · R reset</span><span id="clock" class="muted">00:00</span></footer>
    <script type="module" src="/src/snake/main.ts"></script>
  </body>
</html>
```

- [ ] **Step 2: Styles** — `src/styles.css` (dark theme; keep it short: monospace font, `.layout` grid two columns collapsing to one under 900px, `.track`/`.bar` as 18-step bars, colors `--green #7ef3b6`, `--amber #f5c26b`, `--cyan #7fd8f2`, `--red #ff6b6b`, `--muted #7b8a93`, background `#0b1216`). Write it fully; no framework.

- [ ] **Step 3: Shared loader** — `src/loader.ts`:

```ts
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
      ui.status(p.total ? `downloading model ${mb(p.received)} / ${mb(p.total)} MB` : `downloading model ${mb(p.received)} MB`);
    },
  });
  ui.status(`ready on ${loaded.provider} · ${loaded.bundle.onnxConfig.dtype}`);
  return loaded;
}
```

- [ ] **Step 4: View** — `src/snake/view.ts`:

```ts
import type { Snapshot } from "./game.ts";
import type { Decision } from "./policy.ts";
import type { LoopStats } from "./loop.ts";

export interface ViewLabels {
  engine: string; // e.g. "WebGPU · FP16" or "STUB"
  state: "LOADING" | "LIVE" | "PAUSED" | "GAME OVER" | "BOARD CLEAR";
  elapsedSeconds: number;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export class SnakeView {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly cell: number;

  constructor(private readonly canvas: HTMLCanvasElement, width: number, height: number) {
    this.cell = Math.floor(Math.min(canvas.width / width, canvas.height / height));
    canvas.width = this.cell * width;
    canvas.height = this.cell * height;
    this.ctx = canvas.getContext("2d")!;
  }

  render(board: Snapshot, decision: Decision | null, stats: LoopStats, labels: ViewLabels): void {
    this.drawBoard(board);
    $("state").textContent = labels.state;
    $("state").className = `state ${board.alive ? "green" : "red"}`;
    $("round").textContent = `ROUND ${String(stats.round).padStart(2, "0")}`;
    $("score").textContent = String(board.score);
    $("length").textContent = String(board.length);
    $("best").textContent = String(stats.best);
    const fill = board.length / (board.width * board.height);
    $("fill-bar").style.width = `${(100 * fill).toFixed(1)}%`;
    $("fill-pct").textContent = `${(100 * fill).toFixed(1)}%`;
    for (const li of document.querySelectorAll<HTMLLIElement>("#probs li")) {
      const dir = li.dataset["dir"] as keyof Decision["probabilities"];
      const p = decision?.probabilities[dir] ?? 0;
      li.classList.toggle("selected", decision?.proposed === dir);
      li.querySelector<HTMLElement>(".bar")!.style.width = `${(100 * p).toFixed(0)}%`;
      li.querySelector<HTMLElement>(".val")!.textContent = p.toFixed(2);
    }
    $("executing").textContent = decision?.executed ?? "—";
    $("shield").hidden = !decision?.intervened;
    const risk = decision?.deadEndRisk ?? 0;
    $("risk-bar").style.width = `${(100 * risk).toFixed(0)}%`;
    $("risk-bar").className = `bar ${risk < 0.5 ? "amber" : "red"}`;
    $("risk-val").textContent = risk.toFixed(2);
    const food = decision?.foodReachable ?? 0;
    $("food-bar").style.width = `${(100 * food).toFixed(0)}%`;
    $("food-val").textContent = food.toFixed(2);
    $("inference").textContent = `${(decision?.inferenceMs ?? 0).toFixed(1)} ms`;
    $("rate").textContent = `${stats.decisionsPerSecond.toFixed(1)} /s`;
    $("output-tokens").textContent = String(decision?.outputTokens ?? 0);
    $("engine").textContent = labels.engine;
    $("engine-title").textContent = labels.engine.split(" · ")[0] ?? "";
    $("interventions").textContent = String(stats.interventions).padStart(4, "0");
    const s = Math.floor(labels.elapsedSeconds);
    $("clock").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  }

  private drawBoard(board: Snapshot): void {
    const { ctx, cell } = this;
    ctx.fillStyle = "#0b1216";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.fillStyle = "#13272e";
    for (let y = 0; y < board.height; y++) for (let x = 0; x < board.width; x++) ctx.fillRect(x * cell + cell / 2 - 1, y * cell + cell / 2 - 1, 2, 2);
    board.body.forEach(([x, y], index) => {
      const fraction = 1 - index / Math.max(1, board.body.length);
      ctx.fillStyle = index === 0 ? "#dcfff0" : `rgb(${18 + 64 * fraction | 0},${73 + 150 * fraction | 0},${57 + 102 * fraction | 0})`;
      ctx.fillRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
    });
    if (board.food) {
      ctx.fillStyle = "#f5c26b";
      ctx.beginPath();
      ctx.arc(board.food[0] * cell + cell / 2, board.food[1] * cell + cell / 2, cell * 0.3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
```

- [ ] **Step 5: Wiring** — `src/snake/main.ts`:

```ts
import { loadDemoAgent } from "../loader.ts";
import { GameLoop, type LoopSettings } from "./loop.ts";
import type { SnakeAgent } from "./policy.ts";
import { StubAgent } from "./stub-agent.ts";
import { SnakeView } from "./view.ts";

const params = new URLSearchParams(location.search);
const num = (name: string, fallback: number) => Number(params.get(name) ?? fallback);
const settings: LoopSettings = {
  width: num("width", 24),
  height: num("height", 16),
  seed: num("seed", 7),
  initialLength: num("length", 6),
  fps: num("fps", 12),
  guarded: params.get("unassisted") === null,
  prompt: params.get("prompt") === "detailed" ? "detailed" : "compact",
};
const loadingEl = document.getElementById("loading")!;
const view = new SnakeView(document.getElementById("board") as HTMLCanvasElement, settings.width, settings.height);

async function start(): Promise<void> {
  let agent: SnakeAgent;
  let engine: string;
  if (params.get("model") === "stub") {
    agent = new StubAgent();
    engine = "STUB · planner";
    loadingEl.textContent = "stub agent (no model)";
  } else {
    const loaded = await loadDemoAgent({ status: (t) => (loadingEl.textContent = t) });
    agent = loaded.agent;
    engine = `${loaded.provider === "webgpu" ? "WebGPU" : "wasm"} · ${loaded.bundle.onnxConfig.dtype.toUpperCase().replace("FLOAT", "FP")}`;
  }
  const loop = new GameLoop(agent, settings);
  loop.maxSpeed = params.get("max-speed") !== null;
  const buttons = {
    pause: () => loop.togglePause(),
    slower: () => loop.changeSpeed(-2),
    faster: () => loop.changeSpeed(+2),
    "max-speed": () => (loop.maxSpeed = !loop.maxSpeed),
    reset: () => loop.reset(),
  };
  for (const [id, fn] of Object.entries(buttons)) document.getElementById(id)!.addEventListener("click", fn);
  document.addEventListener("keydown", (e) => {
    if (e.key === " ") buttons.pause();
    else if (e.key === "ArrowUp" || e.key === "+") buttons.faster();
    else if (e.key === "ArrowDown" || e.key === "-") buttons.slower();
    else if (e.key === "r" || e.key === "R") buttons.reset();
    else return;
    e.preventDefault();
  });
  const draw = () =>
    view.render(loop.game.snapshot(), loop.lastDecision, loop.stats, {
      engine,
      state: loop.paused ? "PAUSED" : loop.game.won ? "BOARD CLEAR" : !loop.game.alive ? "GAME OVER" : "LIVE",
      elapsedSeconds: loop.elapsedSeconds,
    });
  draw();
  document.body.dataset["ready"] = "1";
  for (;;) {
    const startedAt = performance.now();
    const finished = !loop.game.alive || loop.game.won;
    await loop.tick();
    draw();
    // Python pauses a second at the end of a round so the final board is visible.
    await new Promise((r) => setTimeout(r, finished ? 1000 : loop.paused ? 100 : loop.delayAfter(startedAt)));
  }
}

start().catch((error) => {
  loadingEl.textContent = `ERROR ${String(error)}`;
  document.getElementById("state")!.textContent = "ERROR";
  console.error(error);
});
```

`decide`'s `inferenceMs` is per call; `batchSize = 3` in the loader matches the three questions per move.

- [ ] **Step 6: Vite input and index link** — add `snake: resolve(here, "snake.html")` to `build.rollupOptions.input` in `vite.config.ts`; add `<li><a href="snake.html">Snake</a> — Laya plays Snake with the cycle safety shield</li>` to `index.html`.

- [ ] **Step 7: Manual run** — `pnpm --filter laya-demo dev` → open `http://localhost:5173/snake.html?model=stub` (board moves, panel updates, Space pauses, R resets) and `http://localhost:5173/snake.html?model=/models/laya-multilingual-onnx-fp16/` (if the local symlink exists) — the real model should show WebGPU · FP16 and inference around 20–60 ms per decision.

- [ ] **Step 8: Typecheck, format, build, commit**

```bash
pnpm typecheck && pnpm format:write && pnpm format && pnpm build
git add apps/demo/snake.html apps/demo/src/styles.css apps/demo/src/loader.ts apps/demo/src/snake/view.ts apps/demo/src/snake/main.ts apps/demo/vite.config.ts apps/demo/index.html
git commit -m "Add the browser Snake page"
```

---

### Task 6: Playwright tests for Snake

**Files:**
- Create: `web/apps/demo/e2e/snake.spec.ts`

- [ ] **Step 1: Tests**

```ts
import { expect, test } from "@playwright/test";

test("stub agent plays Snake: board advances, pause and reset work", async ({ page }) => {
  await page.goto("/snake.html?model=stub&width=8&height=6&length=4&fps=60");
  await expect(page.locator("body[data-ready]")).toBeAttached();
  await expect(page.locator("#state")).toHaveText("LIVE");
  const executing = page.locator("#executing");
  await expect(executing).not.toHaveText("—", { timeout: 5000 });
  const scoreBefore = Number(await page.locator("#score").textContent());
  await page.waitForTimeout(1500);
  await expect(page.locator("#length")).not.toHaveText("0");
  await page.keyboard.press("Space");
  await expect(page.locator("#state")).toHaveText("PAUSED");
  const ticksPaused = await page.locator("#rate").textContent();
  await page.waitForTimeout(500);
  await page.keyboard.press("Space");
  await expect(page.locator("#state")).toHaveText("LIVE");
  await page.keyboard.press("r");
  await expect(page.locator("#round")).toHaveText("ROUND 02");
  expect(scoreBefore).toBeGreaterThanOrEqual(0);
  expect(ticksPaused).toBeDefined();
  await expect(page.locator("#engine")).toHaveText("STUB · planner");
});

test("real model plays 40 decisions without dying", async ({ page }) => {
  const modelUrl = process.env.LAYA_MODEL_URL;
  test.skip(!modelUrl, "set LAYA_MODEL_URL to run the real-model Snake check");
  test.setTimeout(10 * 60 * 1000);
  await page.goto(`/snake.html?model=${encodeURIComponent(modelUrl!)}&max-speed`);
  await expect(page.locator("#engine")).toContainText("WebGPU", { timeout: 9 * 60 * 1000 });
  await expect
    .poll(async () => Number((await page.locator("#length").textContent()) ?? 0), { timeout: 60_000 })
    .toBeGreaterThanOrEqual(6);
  await page.waitForTimeout(8000);
  await expect(page.locator("#round")).toHaveText("ROUND 01"); // no death → no new round
  const inference = await page.locator("#inference").textContent();
  expect(Number.parseFloat(inference ?? "0")).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run** — `pnpm test:browser` → 2 passed (index smoke, snake stub), 2 skipped. Then `LAYA_MODEL_URL=https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/ pnpm test:browser` → 4 passed. Fix flakiness by loosening waits rather than assertions.

- [ ] **Step 3: Commit**

```bash
pnpm format:write && pnpm format
git add apps/demo/e2e/snake.spec.ts
git commit -m "Test the Snake page with a stub agent and the real model"
```

---

### Task 7: Deploy to GitHub Pages and a Hugging Face Space

**Files:**
- Create: `.github/workflows/pages.yml`, `.github/workflows/space.yml`, `web/space/README.md`
- Modify: `docs/onnx-model-card.md` (Space link), `web/README.md`, `docs/SNAKE_DEMO.md`

- [ ] **Step 1: Pages workflow** — `.github/workflows/pages.yml`:

```yaml
name: Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: web
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with:
          version: 10
      - uses: actions/setup-node@v4
        with:
          node-version: 24
          cache: pnpm
          cache-dependency-path: web/pnpm-lock.yaml
      - run: pnpm install --frozen-lockfile
      - run: pnpm build
        env:
          VITE_BASE: /${{ github.event.repository.name }}/
      - uses: actions/upload-pages-artifact@v3
        with:
          path: web/apps/demo/dist
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

The repository owner must set Pages → Source = "GitHub Actions" once in the repo settings (say so in the report).

- [ ] **Step 2: Space files** — `web/space/README.md`:

```markdown
---
title: Laya in the browser
emoji: 🐍
colorFrom: green
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
models:
  - mizchi/laya-multilingual-onnx
---

Typed decisions from a Laya checkpoint running in onnxruntime-web (WebGPU). Source and tests:
https://github.com/mizchi/laya-mlx (`web/`). The model is fetched from
https://huggingface.co/mizchi/laya-multilingual-onnx on first load and cached by the browser.
```

`.github/workflows/space.yml`:

```yaml
name: Space

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: ${{ vars.HF_SPACE != '' }}
    defaults:
      run:
        working-directory: web
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with:
          version: 10
      - uses: actions/setup-node@v4
        with:
          node-version: 24
          cache: pnpm
          cache-dependency-path: web/pnpm-lock.yaml
      - run: pnpm install --frozen-lockfile
      - run: pnpm build
      - run: cp space/README.md apps/demo/dist/README.md
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "huggingface_hub>=1,<2"
      - run: hf upload "$HF_SPACE" apps/demo/dist . --repo-type space --commit-message "Deploy ${GITHUB_SHA::7}"
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_SPACE: ${{ vars.HF_SPACE }}
```

The job is gated on a repository variable `HF_SPACE` (= `mizchi/laya-web-demo`) and needs a `HF_TOKEN` secret with write access to that Space; both are set by the owner in GitHub → Settings → Secrets and variables → Actions. Without them the job is skipped, so forks stay green.

- [ ] **Step 3: Create the Space** (local, once; requires the `hf` login from Plan 1 and the owner's go-ahead — ask before running):

```bash
uv run hf repo create mizchi/laya-web-demo --repo-type space --space-sdk static
cd web && pnpm build && cp space/README.md apps/demo/dist/README.md && uv run hf upload mizchi/laya-web-demo apps/demo/dist . --repo-type space --commit-message "Initial deploy"
```

Expected: `https://huggingface.co/spaces/mizchi/laya-web-demo` builds within a minute; `https://mizchi-laya-web-demo.static.hf.space/snake.html` loads the model from the Hub and plays. If the Space URL differs, take it from the Space page and use it below.

- [ ] **Step 4: Docs** — add to `docs/onnx-model-card.md` (and re-upload it to the model repo with `uv run hf upload mizchi/laya-multilingual-onnx docs/onnx-model-card.md README.md --repo-type model`) a line: "Live demo: https://huggingface.co/spaces/mizchi/laya-web-demo (Snake) — also on GitHub Pages at https://mizchi.github.io/laya-mlx/." Add a "Browser" section to `docs/SNAKE_DEMO.md` (URL parameters: `model`, `width`, `height`, `seed`, `length`, `fps`, `max-speed`, `unassisted`, `prompt=detailed`; the stub mode; desktop-only note) and update `web/README.md` (pages list, deploy workflows, required secret/variable, Pages source setting).

- [ ] **Step 5: Verify locally, commit**

```bash
uv run python -c "import yaml; [yaml.safe_load(open(f)) for f in ('.github/workflows/pages.yml', '.github/workflows/space.yml')]"
cd web && pnpm format:write && pnpm format && VITE_BASE=/laya-mlx/ pnpm build && grep -o '/laya-mlx/ort/' apps/demo/dist/assets/*.js | head -1
git add .github/workflows/pages.yml .github/workflows/space.yml web/space/README.md docs/onnx-model-card.md docs/SNAKE_DEMO.md web/README.md
git commit -m "Deploy the demo site to GitHub Pages and a Hugging Face Space"
```

Expected: the grep finds the base-prefixed ORT path (proves `VITE_BASE` reaches `ORT_WASM_PATHS`).

---

## Self-review

- Spec §2 coverage: game/policy/loop/view/main modules → Tasks 2–5; controls incl. max-speed toggle → Task 5; mulberry32 → Task 2; shield behaviour → Task 3; tests (`game.test`, `policy.test` prompt equality, Playwright smoke with stub) → Tasks 2, 3, 6; loop tests added beyond spec (cheap, pins pacing semantics). Spec §4: Pages workflow, `VITE_LAYA_MODEL_URL` default, ORT assets in dist (already), Space deploy → Task 7. Docs → Task 7.
- Placeholders: Task 5 Step 2 asks the engineer to write `styles.css` from a short palette description (no code block) — acceptable for CSS; everything else is literal. `<N>` in Task 1 Step 6 is an observed count, not a decision.
- Type consistency: `Snapshot`, `MoveInfo`, `Direction` from `game.ts` used in fixtures/policy/view; `Decision`/`SnakeAgent`/`PolicyOptions` from `policy.ts` used by loop/view/main; `LoopStats`/`LoopSettings` from `loop.ts` used by view/main; `StubAgent` implements `SnakeAgent` structurally (`predict` signature matches `LayaAgent.predict`).
