# Browser runtime `@laya-mlx/web` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A TypeScript library that runs the ONNX export of a Laya checkpoint in the browser (onnxruntime-web, WebGPU with wasm fallback) and returns exactly the JSON the Python `Agent.predict` returns, verified against fixtures dumped from Python.

**Architecture:** Pure modules (`tokenizer`, `prompt`, `calibration`, `agent`) reproduce `laya_mlx/common.py` and `laya_mlx/agent.py` token-for-token and are tested in Node against `web/fixtures/parity-multilingual.json`. One I/O module (`session`) downloads the bundle published by `laya-mlx export-onnx`, caches `model.onnx` in the Cache API and wraps `InferenceSession`. A Vite multi-page app (`web/apps/demo`) hosts a `parity.html` page used by an env-gated Playwright test with the real model; Snake and Chess pages are added by the next plans.

**Tech Stack:** pnpm workspace, TypeScript 5, Vite 8, vitest 5, @playwright/test 1.63, onnxruntime-web 1.30, @huggingface/tokenizers 0.2, Python (uv) for fixture generation, `hf` CLI for model upload.

Spec: `docs/superpowers/specs/2026-09-20-web-demos-design.md` (sections 1 and 4). Plans 2 and 3 cover Snake and Chess.

---

## File structure

```
benchmarks/web_fixtures.py                 dumps parity fixtures (Python → JSON)
web/
  package.json                             workspace root scripts: typecheck, test, test:browser, format
  pnpm-workspace.yaml
  tsconfig.base.json
  .prettierrc
  fixtures/parity-multilingual.json        committed output of web_fixtures.py
  packages/laya-web/
    package.json                           name @laya-mlx/web, type module
    tsconfig.json
    vitest.config.ts
    src/index.ts                           public exports
    src/types.ts                           contract: questions, answers, config, batch
    src/pyjson.ts                          Python json.dumps-compatible serializer
    src/questions.ts                       toInternal (validation), renderOptions
    src/tokenizer.ts                       LayaTokenizer (added-token + Metaspace split, tokenizers.js pieces)
    src/prompt.ts                          buildPrefix, buildSequence, collate
    src/calibration.ts                     softmax, confidence, tempBucket, formatAnswers
    src/agent.ts                           LayaAgent (predict), Runner interface
    src/session.ts                         loadBundle (fetch + Cache API + progress), OnnxRunner
    test/fixtures.ts                       loads ../../../fixtures/parity-multilingual.json
    test/pyjson.test.ts
    test/questions.test.ts
    test/tokenizer.test.ts
    test/prompt.test.ts
    test/calibration.test.ts
    test/agent.test.ts
  apps/demo/
    package.json
    tsconfig.json
    vite.config.ts                         multi-page (parity.html now; snake/chess later), copies ORT wasm assets
    index.html                             links to the pages
    parity.html                            runs the fixture batches through the real model, prints JSON
    src/parity.ts
    src/model-url.ts                       VITE_LAYA_MODEL_URL with the Hugging Face default
    playwright.config.ts
    e2e/parity.spec.ts                     env-gated real-model parity
.github/workflows/ci.yml                   add a `web` job
.gitignore                                 add node_modules/, playwright artifacts
```

Naming used throughout (keep these exact):

- `InternalQuestion = { t: QuestionType; ins: string; crit: ... }` (mirrors Python's `{"t", "ins", "crit"}`)
- `PreparedItem = { ids: number[]; markers: number[]; qtype: number }`
- `Batch = { rows, length, markers, inputIds: BigInt64Array, attentionMask: BigInt64Array, markerPos: BigInt64Array, markerMask: Uint8Array, qtype: BigInt64Array }`
- `RunnerOutput = { logits: Float32Array; actLogits: Float32Array; rows: number; markers: number; actions: number }`

---

### Task 1: Python fixture dump

**Files:**
- Create: `benchmarks/web_fixtures.py`
- Create: `web/fixtures/parity-multilingual.json` (generated)
- Modify: `.gitignore`

- [ ] **Step 1: Write the dump script**

```python
"""Dump parity fixtures for the browser runtime (web/packages/laya-web).

Every case records the public inputs, the Python-prepared token sequences,
the collated batch, MLX CPU float32 logits and the public `predict` result,
so the TypeScript port can be checked token-for-token without a model.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from laya_mlx import Agent
from laya_mlx.agent import collate_items

from .common import parity_cases

ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_CASES = [
    "  leading  spaces", "a  b", "   ", "x ", " x", "line\nbreak", "tab\there", "\n", "\n\n\n",
    "\nhello", "hello\n", "\t", "\thello", "\r", " b", "1", "。", "<mask>x", "x<mask>y",
    "x <mask> y", "<mask>", "<mask><mask>", "\nx\n", '{"message": "hi  there"}', "▁literal",
    "▁▁x", "<2mass>x", "[@BOS@]x", "<unused0>x", "mixed 日本語 and English  double", "emoji 😀😀",
    "level 0: not urgent", "a.b*c+d?e^f$g{h}(i)|j[k]\\l",
    "Ich wurde zweimal für Rechnung 4411 belastet", "मुझसे इनवॉइस 4411", "С меня дважды",
    "発票4411被重复扣款，请今天退款。", "[MASK] <mask> hello [MASK] <mask>", "",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="aac6fef/laya-multilingual-mlx")
    parser.add_argument("--output", type=Path, default=ROOT / "web/fixtures/parity-multilingual.json")
    args = parser.parse_args()
    agent = Agent(args.model, dtype="float32", device="cpu", batch_size=16)
    tok = agent.tok
    cases = []
    for name, state, questions in parity_cases():
        items, _ = agent.prepare(state, questions)
        batch = collate_items(items, tok.pad_token_id)
        logits, act = (np.asarray(x, dtype=np.float32) for x in agent.forward(batch))
        cases.append(
            {
                "name": name,
                "state": state,
                "questions": questions,
                "items": items,
                "batch": {k: v.astype(int).tolist() for k, v in batch.items()},
                "logits": logits.tolist(),
                "act_logits": act.tolist(),
                "result": agent.predict(state, questions),
            }
        )
    fixture = {
        "model": args.model,
        "config": agent.cfg,
        "special_tokens": {
            "cls": tok.cls_token_id,
            "sep": tok.sep_token_id,
            "pad": tok.pad_token_id,
            "mask": tok.mask_token_id,
            "mask_token": tok.mask_token,
        },
        "tokenizer_cases": [{"text": t, "ids": tok(t)["input_ids"]} for t in TOKENIZER_CASES],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"{len(cases)} cases, {sum(len(c['items']) for c in cases)} questions -> {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd /Users/mz/ghq/github.com/mizorewww/laya-mlx && uv run python -m benchmarks.web_fixtures`
Expected: `16 cases, 63 questions -> .../web/fixtures/parity-multilingual.json`

- [ ] **Step 3: Sanity-check the file**

Run: `python3 -c "import json; d=json.load(open('web/fixtures/parity-multilingual.json')); print(len(d['cases']), d['config']['max_len'], d['special_tokens'], len(d['tokenizer_cases']))"`
Expected: `16 1024 {'cls': 2, 'sep': 1, 'pad': 0, 'mask': 4, 'mask_token': '<mask>'} 39`

- [ ] **Step 4: Ignore Node artifacts**

Append to `.gitignore`:

```
node_modules/
web/apps/demo/test-results/
web/apps/demo/playwright-report/
```

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check benchmarks/web_fixtures.py && uv run ruff format benchmarks/web_fixtures.py`
Expected: `All checks passed!`

```bash
git add benchmarks/web_fixtures.py web/fixtures/parity-multilingual.json .gitignore
git commit -m "Dump browser parity fixtures from the MLX runtime"
```

---

### Task 2: Workspace scaffold

**Files:**
- Create: `web/package.json`, `web/pnpm-workspace.yaml`, `web/tsconfig.base.json`, `web/.prettierrc`
- Create: `web/packages/laya-web/package.json`, `web/packages/laya-web/tsconfig.json`, `web/packages/laya-web/vitest.config.ts`, `web/packages/laya-web/src/index.ts`
- Create: `web/packages/laya-web/test/fixtures.ts`

- [ ] **Step 1: Root files**

`web/package.json`:

```json
{
  "name": "laya-web-workspace",
  "private": true,
  "type": "module",
  "packageManager": "pnpm@10.28.2",
  "engines": { "node": ">=24" },
  "scripts": {
    "typecheck": "pnpm -r typecheck",
    "test": "pnpm -r test",
    "test:browser": "pnpm --filter laya-demo test:browser",
    "build": "pnpm --filter laya-demo build",
    "format": "prettier --check .",
    "format:write": "prettier --write ."
  },
  "devDependencies": {
    "prettier": "^3.6.0",
    "typescript": "^5.9.0"
  }
}
```

`web/pnpm-workspace.yaml`:

```yaml
packages:
  - packages/*
  - apps/*
```

`web/tsconfig.base.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "verbatimModuleSyntax": true,
    "resolveJsonModule": true,
    "skipLibCheck": true,
    "noEmit": true
  }
}
```

`web/.prettierrc`:

```json
{ "printWidth": 100, "semi": true, "singleQuote": false }
```

- [ ] **Step 2: Library package**

`web/packages/laya-web/package.json`:

```json
{
  "name": "@laya-mlx/web",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "exports": { ".": "./src/index.ts" },
  "scripts": {
    "typecheck": "tsc -p tsconfig.json",
    "test": "vitest run"
  },
  "dependencies": {
    "@huggingface/tokenizers": "^0.2.0",
    "onnxruntime-web": "^1.30.0"
  },
  "devDependencies": {
    "vitest": "^5.0.0"
  }
}
```

`web/packages/laya-web/tsconfig.json`:

```json
{
  "extends": "../../tsconfig.base.json",
  "include": ["src", "test", "vitest.config.ts"]
}
```

`web/packages/laya-web/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: { include: ["test/**/*.test.ts"], environment: "node" },
});
```

`web/packages/laya-web/src/index.ts`:

```ts
export {};
```

`web/packages/laya-web/test/fixtures.ts`:

```ts
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import type { AgentConfig, PredictResult, Question, State } from "../src/types.ts";

export interface FixtureCase {
  name: string;
  state: State;
  questions: Record<string, Question>;
  items: { ids: number[]; markers: number[]; qtype: number }[];
  batch: {
    input_ids: number[][];
    attention_mask: number[][];
    marker_pos: number[][];
    marker_mask: number[][];
    qtype: number[];
  };
  logits: number[][];
  act_logits: number[][];
  result: PredictResult;
}

export interface Fixture {
  model: string;
  config: AgentConfig;
  special_tokens: { cls: number; sep: number; pad: number; mask: number; mask_token: string };
  tokenizer_cases: { text: string; ids: number[] }[];
  cases: FixtureCase[];
}

const path = fileURLToPath(new URL("../../../fixtures/parity-multilingual.json", import.meta.url));
export const fixture: Fixture = JSON.parse(readFileSync(path, "utf8"));
```

- [ ] **Step 3: Install and verify the toolchain**

Run: `cd /Users/mz/ghq/github.com/mizorewww/laya-mlx/web && pnpm install && pnpm typecheck`
Expected: install succeeds; `tsc` reports errors only about the missing `../src/types.ts` (created in Task 3). That is fine for now.

- [ ] **Step 4: Commit**

```bash
git add web/package.json web/pnpm-workspace.yaml web/pnpm-lock.yaml web/tsconfig.base.json web/.prettierrc web/packages/laya-web
git commit -m "Scaffold the web workspace and @laya-mlx/web package"
```

---

### Task 3: Contract types and question validation

**Files:**
- Create: `web/packages/laya-web/src/types.ts`
- Create: `web/packages/laya-web/src/pyjson.ts`
- Create: `web/packages/laya-web/src/questions.ts`
- Test: `web/packages/laya-web/test/pyjson.test.ts`, `web/packages/laya-web/test/questions.test.ts`

- [ ] **Step 1: Types**

`web/packages/laya-web/src/types.ts`:

```ts
export type QuestionType = "choice" | "score" | "noul";
export const QTYPES: Record<QuestionType, number> = { choice: 0, score: 1, noul: 2 };
export const QTYPE_NAMES: QuestionType[] = ["choice", "score", "noul"];

export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };
export type Criterion = Json;
export type State = string | Json[] | { [key: string]: Json };

export interface ChoiceQuestion {
  type: "choice";
  instructions: string | Json;
  criteria: string[] | Record<string, Criterion>;
}
export interface ScoreQuestion {
  type: "score";
  instructions: string | Json;
  criteria: Criterion[];
}
export interface NoulQuestion {
  type: "noul";
  instructions: string | Json;
  criteria?: { false?: Criterion; true?: Criterion } | null;
}
export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;

/** Mirrors Python `Agent._to_internal` output: {"t", "ins", "crit"}. */
export type InternalQuestion =
  | { t: "choice"; ins: string; crit: Record<string, Criterion | null> }
  | { t: "score"; ins: string; crit: Criterion[] }
  | { t: "noul"; ins: string; crit: { false?: Criterion; true?: Criterion } | null };

export interface AgentConfig {
  encoder: string;
  head_layers: number;
  max_len: number;
  head_max_len: number;
  max_prefixes?: number;
  act_costs?: Record<string, number>;
  temperature: number[];
  temperature_by_options: Record<string, number>;
  [extra: string]: Json | undefined;
}

export interface PreparedItem {
  ids: number[];
  markers: number[];
  qtype: number;
}

export interface Batch {
  rows: number;
  length: number;
  markers: number;
  inputIds: BigInt64Array;
  attentionMask: BigInt64Array;
  markerPos: BigInt64Array;
  markerMask: Uint8Array;
  qtype: BigInt64Array;
}

export interface RunnerOutput {
  logits: Float32Array;
  actLogits: Float32Array;
  rows: number;
  markers: number;
  actions: number;
}

export interface AnswerAction {
  act_probability: number;
}
export interface ChoiceAnswer {
  type: "choice";
  confidence: number;
  action: AnswerAction;
  choice: string;
  probabilities: Record<string, number>;
}
export interface ScoreAnswer {
  type: "score";
  confidence: number;
  action: AnswerAction;
  score: number;
  legend: Record<string, Criterion>;
  probabilities: Record<string, number>;
}
export interface NoulAnswer {
  type: "noul";
  confidence: number;
  action: AnswerAction;
  noul: number;
}
export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface PredictResult {
  model: "laya-rl-agent";
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: 0 };
}
```

- [ ] **Step 2: Failing tests for the Python-compatible JSON serializer**

`web/packages/laya-web/test/pyjson.test.ts`:

```ts
import { describe, expect, it } from "vitest";

import { pyJson } from "../src/pyjson.ts";

describe("pyJson", () => {
  it("uses Python's default separators", () => {
    expect(pyJson({ task: "choose department", n: [1, 2] })).toBe(
      '{"task": "choose department", "n": [1, 2]}',
    );
  });
  it("keeps non-ASCII when ensureAscii is false and escapes it otherwise", () => {
    expect(pyJson({ message: "請求書" }, { ensureAscii: false })).toBe('{"message": "請求書"}');
    expect(pyJson({ message: "請求書" })).toBe('{"message": "\\u8acb\\u6c42\\u66f8"}');
    expect(pyJson("😀")).toBe('"\\ud83d\\ude00"');
  });
  it("serializes primitives like Python", () => {
    expect(pyJson(true)).toBe("true");
    expect(pyJson(null)).toBe("null");
    expect(pyJson(1.5)).toBe("1.5");
    expect(pyJson("a\"b\n")).toBe('"a\\"b\\n"');
  });
});
```

- [ ] **Step 3: Run to verify failure**

Run: `cd /Users/mz/ghq/github.com/mizorewww/laya-mlx/web && pnpm --filter @laya-mlx/web test -- pyjson`
Expected: FAIL, cannot find module `../src/pyjson.ts`

- [ ] **Step 4: Implement `pyjson.ts`**

```ts
import type { Json } from "./types.ts";

export interface PyJsonOptions {
  /** Python's json.dumps default. false matches `ensure_ascii=False`. */
  ensureAscii?: boolean;
}

/**
 * Serialize like Python's `json.dumps(value, separators=(", ", ": "))`.
 *
 * Differences that cannot be reproduced from JavaScript values are documented
 * rather than hidden: integer-valued floats print as integers (Python prints
 * `1.0`), and integer-like object keys are enumerated first by JavaScript.
 */
export function pyJson(value: Json, options: PyJsonOptions = {}): string {
  return serialize(value, options.ensureAscii ?? true);
}

function serialize(value: Json, ensureAscii: boolean): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new RangeError("Non-finite numbers are not JSON");
    return String(value);
  }
  if (typeof value === "string") return serializeString(value, ensureAscii);
  if (Array.isArray(value)) return "[" + value.map((v) => serialize(v, ensureAscii)).join(", ") + "]";
  return (
    "{" +
    Object.entries(value)
      .map(([k, v]) => serializeString(k, ensureAscii) + ": " + serialize(v, ensureAscii))
      .join(", ") +
    "}"
  );
}

function serializeString(value: string, ensureAscii: boolean): string {
  const escaped = JSON.stringify(value);
  if (!ensureAscii) return escaped;
  let out = "";
  for (let i = 0; i < escaped.length; i++) {
    const code = escaped.charCodeAt(i);
    out += code < 0x80 ? escaped[i] : "\\u" + code.toString(16).padStart(4, "0");
  }
  return out;
}
```

- [ ] **Step 5: Run the pyjson tests**

Run: `pnpm --filter @laya-mlx/web test -- pyjson`
Expected: 3 passed

- [ ] **Step 6: Failing tests for question validation and option rendering**

`web/packages/laya-web/test/questions.test.ts`:

```ts
import { describe, expect, it } from "vitest";

import { renderOptions, toInternal } from "../src/questions.ts";
import { fixture } from "./fixtures.ts";

describe("toInternal", () => {
  it("turns choice label lists into null-valued maps", () => {
    expect(toInternal({ type: "choice", instructions: "Pick", criteria: ["a", "b"] })).toEqual({
      t: "choice",
      ins: "Pick",
      crit: { a: null, b: null },
    });
  });
  it("serializes non-string instructions with Python json.dumps (ASCII escaped)", () => {
    expect(toInternal({ type: "score", instructions: { k: "請" }, criteria: ["x"] }).ins).toBe(
      '{"k": "\\u8acb"}',
    );
  });
  it.each([
    [{ type: "choice", instructions: "x", criteria: ["a", "a"] }, /unique/],
    [{ type: "choice", instructions: "x", criteria: [] }, /nonempty/],
    [{ type: "choice", instructions: "x", criteria: [1] }, /strings/],
    [{ type: "score", instructions: "x", criteria: [] }, /nonempty/],
    [{ type: "noul", instructions: "x", criteria: ["a"] }, /dictionary/],
    [{ type: "other", instructions: "x" }, /Unknown question type/],
    [{ type: "noul" }, /missing instructions/],
  ])("rejects %j", (question, message) => {
    expect(() => toInternal(question as never)).toThrow(message);
  });
});

describe("renderOptions", () => {
  it("renders choice, score and noul like Python", () => {
    expect(
      renderOptions({ t: "choice", ins: "", crit: { billing: "refunds", other: null, zero: 0, empty: "" } }),
    ).toEqual(["billing: refunds", "other", "zero: 0", "empty"]);
    expect(renderOptions({ t: "score", ins: "", crit: ["low", { level: "high" }] })).toEqual([
      "level 0: low",
      'level 1: {"level": "high"}',
    ]);
    expect(renderOptions({ t: "noul", ins: "", crit: null })).toEqual([
      "false: no, the statement does not hold",
      "true: yes, the statement holds",
    ]);
    expect(renderOptions({ t: "noul", ins: "", crit: { true: { reason: "money back" } } })).toEqual([
      "false: no, the statement does not hold",
      'true: {"reason": "money back"}',
    ]);
  });
  it("produces one option per marker for every fixture question", () => {
    for (const c of fixture.cases) {
      const questions = Object.values(c.questions);
      questions.forEach((q, i) => {
        expect(renderOptions(toInternal(q)).length, `${c.name}#${i}`).toBe(c.items[i]!.markers.length);
      });
    }
  });
});
```

- [ ] **Step 7: Run to verify failure**

Run: `pnpm --filter @laya-mlx/web test -- questions`
Expected: FAIL, cannot find module `../src/questions.ts`

- [ ] **Step 8: Implement `questions.ts`**

```ts
import { pyJson } from "./pyjson.ts";
import type { Criterion, InternalQuestion, Json, Question, QuestionType } from "./types.ts";
import { QTYPES } from "./types.ts";

const isPlainObject = (v: unknown): v is Record<string, Json> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** Port of Python `Agent._to_internal`, including its error messages. */
export function toInternal(question: Question): InternalQuestion {
  if (!isPlainObject(question)) throw new Error("Each question must be a dictionary");
  const kind = question.type as QuestionType;
  if (!(kind in QTYPES)) {
    throw new Error(`Unknown question type ${JSON.stringify(kind)}; expected choice, score, or noul`);
  }
  if (!("instructions" in question)) throw new Error("Question is missing instructions");
  const raw = question.instructions;
  const ins = typeof raw === "string" ? raw : pyJson(raw as Json);
  let criteria: unknown = (question as { criteria?: unknown }).criteria;
  if (kind === "choice") {
    if (Array.isArray(criteria)) {
      if (!criteria.every((c) => typeof c === "string")) throw new Error("Choice labels must be strings");
      if (new Set(criteria).size !== criteria.length) throw new Error("Choice labels must be unique");
      criteria = Object.fromEntries((criteria as string[]).map((c) => [c, null]));
    }
    if (!isPlainObject(criteria) || Object.keys(criteria).length === 0) {
      throw new Error("Choice criteria must be a nonempty dictionary or list");
    }
    return { t: "choice", ins, crit: criteria as Record<string, Criterion | null> };
  }
  if (kind === "score") {
    if (!Array.isArray(criteria) || criteria.length === 0) {
      throw new Error("Score criteria must be a nonempty list");
    }
    return { t: "score", ins, crit: criteria as Criterion[] };
  }
  if (criteria !== undefined && criteria !== null && !isPlainObject(criteria)) {
    throw new Error("Noul criteria must be a dictionary with false/true descriptions");
  }
  return { t: "noul", ins, crit: (criteria as { false?: Criterion; true?: Criterion } | null | undefined) ?? null };
}

/** Python `render_criterion`: strings pass through, structured values become compact JSON. */
export function renderCriterion(value: Criterion): string {
  return typeof value === "string" ? value : pyJson(value, { ensureAscii: false });
}

const isBlank = (v: Criterion | null | undefined) => v === null || v === undefined || v === "";

/** Python `render_options`: option texts in label-index order; noul is always [false, true]. */
export function renderOptions(q: InternalQuestion): string[] {
  if (q.t === "choice") {
    return Object.entries(q.crit).map(([k, v]) => (isBlank(v) ? k : `${k}: ${renderCriterion(v!)}`));
  }
  if (q.t === "score") return q.crit.map((c, i) => `level ${i}: ${renderCriterion(c)}`);
  const crit = q.crit ?? {};
  return [
    "false: " + (isBlank(crit.false) ? "no, the statement does not hold" : renderCriterion(crit.false!)),
    "true: " + (isBlank(crit.true) ? "yes, the statement holds" : renderCriterion(crit.true!)),
  ];
}
```

- [ ] **Step 9: Run the tests**

Run: `pnpm --filter @laya-mlx/web test -- questions`
Expected: all passed (the fixture loop covers 63 questions)

- [ ] **Step 10: Typecheck, format, commit**

Run: `pnpm typecheck && pnpm format:write`
Expected: no type errors

```bash
git add web/packages/laya-web/src web/packages/laya-web/test
git commit -m "Add question contract, validation and Python-compatible JSON"
```

---

### Task 4: Tokenizer with Rust-compatible splitting

**Files:**
- Create: `web/packages/laya-web/src/tokenizer.ts`
- Test: `web/packages/laya-web/test/tokenizer.test.ts`

Background: `@huggingface/tokenizers` matches added tokens (`▁▁`, `\n`, …) against the *normalized* text and merges consecutive `▁` when splitting, while the Rust `tokenizers` crate (what Python uses) matches added tokens on the raw text and splits Metaspace with `MergedWithNext`. We therefore split the raw text ourselves and only hand `▁`-prefixed pieces without added tokens to the library. Verified on 38 edge strings before this plan was written.

- [ ] **Step 1: Failing test**

`web/packages/laya-web/test/tokenizer.test.ts`:

```ts
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { LayaTokenizer } from "../src/tokenizer.ts";
import { fixture } from "./fixtures.ts";

const dir = process.env.LAYA_TOKENIZER_DIR;

describe.skipIf(!dir)("LayaTokenizer (needs LAYA_TOKENIZER_DIR=<bundle>/tokenizer)", () => {
  const tokenizer = new LayaTokenizer(
    JSON.parse(readFileSync(`${dir}/tokenizer.json`, "utf8")),
    JSON.parse(readFileSync(`${dir}/tokenizer_config.json`, "utf8")),
  );
  it("exposes the special token ids the checkpoint declares", () => {
    expect(tokenizer.clsTokenId).toBe(fixture.special_tokens.cls);
    expect(tokenizer.sepTokenId).toBe(fixture.special_tokens.sep);
    expect(tokenizer.padTokenId).toBe(fixture.special_tokens.pad);
    expect(tokenizer.maskTokenId).toBe(fixture.special_tokens.mask);
    expect(tokenizer.maskToken).toBe(fixture.special_tokens.mask_token);
  });
  it.each(fixture.tokenizer_cases)("encodes $text like Rust tokenizers", ({ text, ids }) => {
    expect(tokenizer.encode(text)).toEqual(ids);
  });
});
```

The tokenizer files are not committed (they belong to the model bundle), so this test and the ones in Tasks 5 and 6 read them from `LAYA_TOKENIZER_DIR`. Use the export made earlier: `/private/tmp/claude-501/-Users-mz-ghq-github-com-mizorewww-laya-mlx/17180354-ee76-45e5-90b7-00371d37542e/scratchpad/onnx-multilingual-float16/tokenizer`, or produce one with `uv run laya-mlx export-onnx --model aac6fef/laya-multilingual-mlx --dtype float16 --output models/laya-multilingual-onnx-fp16` and point at `models/laya-multilingual-onnx-fp16/tokenizer`.

- [ ] **Step 2: Run to verify failure**

Run: `LAYA_TOKENIZER_DIR=<bundle>/tokenizer pnpm --filter @laya-mlx/web test -- tokenizer`
Expected: FAIL, cannot find module `../src/tokenizer.ts`

- [ ] **Step 3: Implement `tokenizer.ts`**

```ts
import { Tokenizer } from "@huggingface/tokenizers";

interface AddedToken {
  id: number;
  content: string;
  lstrip: boolean;
  rstrip: boolean;
}

interface TokenizerJson {
  added_tokens: AddedToken[];
  normalizer: { type: string } | null;
  pre_tokenizer: { type: string; replacement?: string; prepend_scheme?: string; split?: boolean } | null;
}

interface TokenizerConfig {
  cls_token?: string | { content: string };
  sep_token?: string | { content: string };
  pad_token?: string | { content: string };
  mask_token?: string | { content: string };
}

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Encoder matching the Rust `tokenizers` crate for Metaspace BPE tokenizers.
 *
 * Added tokens are matched on the raw text, longest first, honoring lstrip/rstrip.
 * Each remaining segment gets the checkpoint's normalizer (" " -> "▁") and the
 * Metaspace `always` prepend, then is split so every piece starts with "▁"
 * (Rust's MergedWithNext). Pieces contain neither added tokens nor consecutive
 * "▁", which is where tokenizers.js diverges from Rust.
 */
export class LayaTokenizer {
  readonly clsToken: string;
  readonly sepToken: string;
  readonly padToken: string;
  readonly maskToken: string;
  readonly clsTokenId: number;
  readonly sepTokenId: number;
  readonly padTokenId: number;
  readonly maskTokenId: number;
  private readonly inner: Tokenizer;
  private readonly addedPattern: RegExp;
  private readonly addedIds: Map<string, number>;
  private readonly replacement: string;

  constructor(tokenizerJson: TokenizerJson, tokenizerConfig: TokenizerConfig) {
    const pre = tokenizerJson.pre_tokenizer;
    if (pre?.type !== "Metaspace" || pre.prepend_scheme !== "always" || pre.split !== true) {
      throw new Error("LayaTokenizer supports Metaspace(prepend_scheme=always, split=true) tokenizers");
    }
    this.replacement = pre.replacement ?? "▁";
    this.inner = new Tokenizer(tokenizerJson as never, tokenizerConfig as never);
    const added = [...tokenizerJson.added_tokens].sort((a, b) => b.content.length - a.content.length);
    this.addedIds = new Map(added.map((a) => [a.content, a.id]));
    this.addedPattern = new RegExp(
      added
        .map((a) => (a.lstrip ? "\\s*" : "") + "(" + escapeRegExp(a.content) + ")" + (a.rstrip ? "\\s*" : ""))
        .join("|"),
      "gu",
    );
    const special = (name: keyof TokenizerConfig): [string, number] => {
      const raw = tokenizerConfig[name];
      const content = typeof raw === "string" ? raw : raw?.content;
      const id = content === undefined ? undefined : this.addedIds.get(content);
      if (content === undefined || id === undefined) throw new Error(`Tokenizer is missing a valid ${name}`);
      return [content, id];
    };
    [this.clsToken, this.clsTokenId] = special("cls_token");
    [this.sepToken, this.sepTokenId] = special("sep_token");
    [this.padToken, this.padTokenId] = special("pad_token");
    [this.maskToken, this.maskTokenId] = special("mask_token");
  }

  /** Token ids without special tokens, equal to Python `tok(text, add_special_tokens=False)`. */
  encode(text: string): number[] {
    const ids: number[] = [];
    let last = 0;
    for (const match of text.matchAll(this.addedPattern)) {
      ids.push(...this.encodeSegment(text.slice(last, match.index)));
      const content = match.slice(1).find((group) => group !== undefined)!;
      ids.push(this.addedIds.get(content)!);
      last = match.index + match[0].length;
    }
    ids.push(...this.encodeSegment(text.slice(last)));
    return ids;
  }

  private encodeSegment(segment: string): number[] {
    if (segment === "") return [];
    let normalized = segment.replaceAll(" ", this.replacement);
    if (!normalized.startsWith(this.replacement)) normalized = this.replacement + normalized;
    const pieces = normalized.match(new RegExp(`${this.replacement}[^${this.replacement}]*`, "gu")) ?? [];
    const ids: number[] = [];
    for (const piece of pieces) {
      ids.push(...this.inner.encode(piece, { add_special_tokens: false }).ids);
    }
    return ids;
  }
}
```

- [ ] **Step 4: Run the tests**

Run: `LAYA_TOKENIZER_DIR=<bundle>/tokenizer pnpm --filter @laya-mlx/web test -- tokenizer`
Expected: 40 passed (5 special ids in one test + 39 encode cases)

- [ ] **Step 5: Typecheck, format, commit**

Run: `pnpm typecheck && pnpm format:write`

```bash
git add web/packages/laya-web/src/tokenizer.ts web/packages/laya-web/test/tokenizer.test.ts
git commit -m "Add a Rust-compatible tokenizer wrapper for Metaspace checkpoints"
```

---

### Task 5: Prompt construction and collation

**Files:**
- Create: `web/packages/laya-web/src/prompt.ts`
- Test: `web/packages/laya-web/test/prompt.test.ts`

- [ ] **Step 1: Failing test**

`web/packages/laya-web/test/prompt.test.ts`:

```ts
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { buildSequence, collate, serializeState } from "../src/prompt.ts";
import { toInternal } from "../src/questions.ts";
import { LayaTokenizer } from "../src/tokenizer.ts";
import { fixture } from "./fixtures.ts";

const dir = process.env.LAYA_TOKENIZER_DIR;

describe("serializeState", () => {
  it("passes strings through and dumps JSON with ensure_ascii=False", () => {
    expect(serializeState("plain")).toBe("plain");
    expect(serializeState({ message: "請求書", n: 1 })).toBe('{"message": "請求書", "n": 1}');
    expect(serializeState([{ role: "user", content: "hi" }])).toBe('[{"role": "user", "content": "hi"}]');
  });
});

describe.skipIf(!dir)("buildSequence and collate (needs LAYA_TOKENIZER_DIR)", () => {
  const tokenizer = new LayaTokenizer(
    JSON.parse(readFileSync(`${dir}/tokenizer.json`, "utf8")),
    JSON.parse(readFileSync(`${dir}/tokenizer_config.json`, "utf8")),
  );
  const { max_len, head_max_len } = fixture.config;

  it.each(fixture.cases.map((c) => [c.name, c] as const))("%s: ids, markers and qtype match Python", (_, c) => {
    Object.values(c.questions).forEach((question, i) => {
      const item = buildSequence(tokenizer, c.state, toInternal(question), max_len, head_max_len);
      expect(item, `${c.name}#${i}`).toEqual(c.items[i]);
    });
  });

  it.each(fixture.cases.map((c) => [c.name, c] as const))("%s: collate matches Python", (_, c) => {
    const batch = collate(c.items, tokenizer.padTokenId);
    expect(batch.rows).toBe(c.batch.input_ids.length);
    expect(batch.length).toBe(c.batch.input_ids[0]!.length);
    expect(batch.markers).toBe(c.batch.marker_pos[0]!.length);
    expect(Array.from(batch.inputIds, Number)).toEqual(c.batch.input_ids.flat());
    expect(Array.from(batch.attentionMask, Number)).toEqual(c.batch.attention_mask.flat());
    expect(Array.from(batch.markerPos, Number)).toEqual(c.batch.marker_pos.flat());
    expect(Array.from(batch.markerMask)).toEqual(c.batch.marker_mask.flat());
    expect(Array.from(batch.qtype, Number)).toEqual(c.batch.qtype);
  });

  it("collate pads markers to at least two slots", () => {
    const batch = collate([{ ids: [2, 5, 1], markers: [1], qtype: 0 }], tokenizer.padTokenId);
    expect(batch.markers).toBe(2);
    expect(Array.from(batch.markerMask)).toEqual([1, 0]);
  });

  it("collate rejects an empty batch", () => {
    expect(() => collate([], 0)).toThrow("Cannot collate an empty batch");
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `LAYA_TOKENIZER_DIR=<bundle>/tokenizer pnpm --filter @laya-mlx/web test -- prompt`
Expected: FAIL, cannot find module `../src/prompt.ts`

- [ ] **Step 3: Implement `prompt.ts`**

```ts
import { pyJson } from "./pyjson.ts";
import { renderOptions } from "./questions.ts";
import type { LayaTokenizer } from "./tokenizer.ts";
import type { Batch, InternalQuestion, PreparedItem, State } from "./types.ts";
import { QTYPES } from "./types.ts";

/** Python `serialize_state`: strings verbatim, otherwise `json.dumps(state, ensure_ascii=False)`. */
export function serializeState(state: State): string {
  return typeof state === "string" ? state : pyJson(state, { ensureAscii: false });
}

/** Python `build_prefix`: [CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]. */
export function buildPrefix(
  tokenizer: LayaTokenizer,
  q: InternalQuestion,
  headMaxLen: number,
): { ids: number[]; markers: number[] } {
  const mask = tokenizer.maskToken;
  const options = renderOptions(q);
  const ins = q.ins.replaceAll(mask, " ");
  let headIds = tokenizer.encode(`${q.t} question: ${ins}`);
  let optIds = options.map((option) => [
    tokenizer.maskTokenId,
    ...tokenizer.encode(" " + option.replaceAll(mask, " ")).slice(0, 48),
  ]);
  const total = (rows: number[][]) => rows.reduce((n, row) => n + row.length, 0);
  let optBudget = headMaxLen - total(optIds);
  if (optBudget < 16) {
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((row) => row.slice(0, per));
    optBudget = headMaxLen - total(optIds);
  }
  headIds = headIds.slice(0, Math.max(8, optBudget));
  const ids = [tokenizer.clsTokenId, ...headIds, tokenizer.sepTokenId];
  const markers: number[] = [];
  for (const row of optIds) {
    markers.push(ids.length);
    ids.push(...row);
  }
  ids.push(tokenizer.sepTokenId);
  return { ids, markers };
}

/** Python `build_sequence`: prefix + state tokens (right-truncated) + [SEP], capped at maxLen. */
export function buildSequence(
  tokenizer: LayaTokenizer,
  state: State,
  q: InternalQuestion,
  maxLen: number,
  headMaxLen: number,
): PreparedItem {
  const prefix = buildPrefix(tokenizer, q, headMaxLen);
  const room = Math.max(0, maxLen - prefix.ids.length - 1);
  const stateIds = tokenizer.encode(serializeState(state).replaceAll(tokenizer.maskToken, " ")).slice(0, room);
  const ids = [...prefix.ids, ...stateIds, tokenizer.sepTokenId].slice(0, maxLen);
  return { ids, markers: prefix.markers.filter((m) => m < maxLen), qtype: QTYPES[q.t] };
}

/** Python `collate_items`: right-padded int64 tensors with at least two marker slots. */
export function collate(items: PreparedItem[], padId: number): Batch {
  if (items.length === 0) throw new Error("Cannot collate an empty batch");
  const rows = items.length;
  const length = Math.max(...items.map((item) => item.ids.length));
  const markers = Math.max(2, ...items.map((item) => item.markers.length));
  const batch: Batch = {
    rows,
    length,
    markers,
    inputIds: new BigInt64Array(rows * length).fill(BigInt(padId)),
    attentionMask: new BigInt64Array(rows * length),
    markerPos: new BigInt64Array(rows * markers),
    markerMask: new Uint8Array(rows * markers),
    qtype: BigInt64Array.from(items, (item) => BigInt(item.qtype)),
  };
  items.forEach((item, row) => {
    item.ids.forEach((id, i) => {
      batch.inputIds[row * length + i] = BigInt(id);
      batch.attentionMask[row * length + i] = 1n;
    });
    item.markers.forEach((position, i) => {
      batch.markerPos[row * markers + i] = BigInt(position);
      batch.markerMask[row * markers + i] = 1;
    });
  });
  return batch;
}
```

- [ ] **Step 4: Run the tests**

Run: `LAYA_TOKENIZER_DIR=<bundle>/tokenizer pnpm --filter @laya-mlx/web test -- prompt`
Expected: all passed. If a case fails, print both id arrays and find the first differing index; the usual culprits are `pyJson` spacing for structured states and the 48-token option cap.

- [ ] **Step 5: Typecheck, format, commit**

```bash
pnpm typecheck && pnpm format:write
git add web/packages/laya-web/src/prompt.ts web/packages/laya-web/test/prompt.test.ts
git commit -m "Port Laya prompt construction and collation to TypeScript"
```

---

### Task 6: Calibration and answer formatting

**Files:**
- Create: `web/packages/laya-web/src/calibration.ts`
- Test: `web/packages/laya-web/test/calibration.test.ts`

- [ ] **Step 1: Failing test**

`web/packages/laya-web/test/calibration.test.ts`:

```ts
import { describe, expect, it } from "vitest";

import { confidenceFromProbs, formatAnswers, softmax, tempBucket } from "../src/calibration.ts";
import { toInternal } from "../src/questions.ts";
import { fixture } from "./fixtures.ts";

describe("helpers", () => {
  it("softmax sums to one", () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b)).toBeCloseTo(1, 12);
    expect(p[2]).toBeGreaterThan(p[1]!);
  });
  it("tempBucket names option-count buckets like Python", () => {
    expect(tempBucket(0, 2)).toBe("choice:2");
    expect(tempBucket(1, 5)).toBe("score:3-5");
    expect(tempBucket(0, 10)).toBe("choice:6-10");
    expect(tempBucket(2, 11)).toBe("noul:11+");
  });
  it("confidence is 1 for a single option and 0 for a uniform distribution", () => {
    expect(confidenceFromProbs([1], 1)).toBe(1);
    expect(confidenceFromProbs([0.25, 0.25, 0.25, 0.25], 4)).toBeCloseTo(0, 12);
  });
});

describe("formatAnswers", () => {
  it.each(fixture.cases.map((c) => [c.name, c] as const))("%s: reproduces Python predict()", (_, c) => {
    const internal = Object.values(c.questions).map(toInternal);
    const result = formatAnswers({
      config: fixture.config,
      questionIds: Object.keys(c.questions),
      internal,
      items: c.items,
      logits: Float32Array.from(c.logits.flat()),
      actLogits: Float32Array.from(c.act_logits.flat()),
      markers: c.logits[0]!.length,
      actions: c.act_logits[0]!.length,
    });
    expect(result).toEqual(c.result);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm --filter @laya-mlx/web test -- calibration`
Expected: FAIL, cannot find module `../src/calibration.ts`

- [ ] **Step 3: Implement `calibration.ts`**

```ts
import type { AgentConfig, Answer, InternalQuestion, PredictResult, PreparedItem } from "./types.ts";
import { QTYPE_NAMES } from "./types.ts";

export function softmax(values: ArrayLike<number>): number[] {
  const z = Array.from(values);
  const max = Math.max(...z);
  const e = z.map((v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

/** Python `temp_bucket`. */
export function tempBucket(qtype: number, k: number): string {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${QTYPE_NAMES[qtype]}:${size}`;
}

/** Python `confidence_from_probs`: 1 - H(p) / log(k), clipped to [0, 1]. */
export function confidenceFromProbs(p: number[], k: number): number {
  if (k < 2) return 1;
  let entropy = 0;
  for (const v of p.slice(0, k)) {
    const c = Math.min(Math.max(v, 1e-12), 1);
    entropy -= v * Math.log(c);
  }
  return Math.min(Math.max(1 - entropy / Math.log(k), 0), 1);
}

/** Python `round(x, 4)` for the values this runtime produces. */
export const round4 = (x: number): number => Number(x.toFixed(4));

export interface FormatInput {
  config: AgentConfig;
  questionIds: string[];
  internal: InternalQuestion[];
  items: PreparedItem[];
  logits: Float32Array;
  actLogits: Float32Array;
  markers: number;
  actions: number;
}

/** The second half of Python `Agent.system_one`: calibrated answers for one batch. */
export function formatAnswers(input: FormatInput): PredictResult {
  const { config, questionIds, internal, items, logits, actLogits, markers, actions } = input;
  for (const value of [...logits, ...actLogits]) {
    if (!Number.isFinite(value)) throw new RangeError("Non-finite model outputs");
  }
  const answers: Record<string, Answer> = {};
  items.forEach((item, row) => {
    const qid = questionIds[row]!;
    const q = internal[row]!;
    const k = item.markers.length;
    const act = softmax(actLogits.subarray(row * actions, (row + 1) * actions));
    const scale = config.temperature_by_options[tempBucket(item.qtype, k)] ?? config.temperature[item.qtype]!;
    const z = Array.from(logits.subarray(row * markers, row * markers + k), (v) => v / Math.max(1e-3, scale));
    const p = softmax(z);
    const base = {
      confidence: round4(confidenceFromProbs(p, k)),
      action: { act_probability: round4(act[0]!) },
    };
    if (q.t === "choice") {
      const labels = Object.keys(q.crit);
      const best = p.indexOf(Math.max(...p));
      answers[qid] = {
        type: "choice",
        ...base,
        choice: labels[best]!,
        probabilities: Object.fromEntries(labels.map((label, i) => [label, round4(p[i]!)])),
      };
    } else if (q.t === "score") {
      answers[qid] = {
        type: "score",
        ...base,
        score: round4(p.reduce((sum, v, i) => sum + i * v, 0)),
        legend: Object.fromEntries(q.crit.map((value, i) => [String(i), value])),
        probabilities: Object.fromEntries(p.map((v, i) => [String(i), round4(v)])),
      };
    } else {
      const pTrue = p[1]!;
      answers[qid] = {
        type: "noul",
        ...base,
        confidence: round4(Math.max(pTrue, 1 - pTrue)),
        noul: round4(pTrue),
      };
    }
  });
  return {
    model: "laya-rl-agent",
    answers,
    usage: { input_tokens: items.reduce((n, item) => n + item.ids.length, 0), output_tokens: 0 },
  };
}
```

Key order matters for `toEqual`? No — vitest `toEqual` ignores key order, but the Python result puts `confidence` and `action` before `choice`; the object spread above keeps that order anyway.

- [ ] **Step 4: Run the tests**

Run: `pnpm --filter @laya-mlx/web test -- calibration`
Expected: all passed. If a probability differs in the fourth decimal, compare `round4` against Python on that value; `Number(x.toFixed(4))` and Python `round` agree except at exact binary ties, which the fixtures do not contain.

- [ ] **Step 5: Typecheck, format, commit**

```bash
pnpm typecheck && pnpm format:write
git add web/packages/laya-web/src/calibration.ts web/packages/laya-web/test/calibration.test.ts
git commit -m "Port Laya calibration and answer formatting to TypeScript"
```

---

### Task 7: Agent with an injectable runner

**Files:**
- Create: `web/packages/laya-web/src/agent.ts`
- Modify: `web/packages/laya-web/src/index.ts`
- Test: `web/packages/laya-web/test/agent.test.ts`

- [ ] **Step 1: Failing test**

`web/packages/laya-web/test/agent.test.ts`:

```ts
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { LayaAgent, type Runner } from "../src/agent.ts";
import { LayaTokenizer } from "../src/tokenizer.ts";
import type { Batch, RunnerOutput } from "../src/types.ts";
import { fixture, type FixtureCase } from "./fixtures.ts";

const dir = process.env.LAYA_TOKENIZER_DIR;

/** Replays the fixture logits for whichever rows of the case the agent asks for. */
class FixtureRunner implements Runner {
  calls: Batch[] = [];
  constructor(private readonly c: FixtureCase) {}
  async run(batch: Batch): Promise<RunnerOutput> {
    this.calls.push(batch);
    const offset = this.calls.slice(0, -1).reduce((n, b) => n + b.rows, 0);
    const markers = this.c.logits[0]!.length;
    const actions = this.c.act_logits[0]!.length;
    return {
      rows: batch.rows,
      markers,
      actions,
      logits: Float32Array.from(this.c.logits.slice(offset, offset + batch.rows).flat()),
      actLogits: Float32Array.from(this.c.act_logits.slice(offset, offset + batch.rows).flat()),
    };
  }
}

describe.skipIf(!dir)("LayaAgent (needs LAYA_TOKENIZER_DIR)", () => {
  const tokenizer = new LayaTokenizer(
    JSON.parse(readFileSync(`${dir}/tokenizer.json`, "utf8")),
    JSON.parse(readFileSync(`${dir}/tokenizer_config.json`, "utf8")),
  );

  it.each(fixture.cases.map((c) => [c.name, c] as const))("%s: predict matches Python", async (_, c) => {
    const runner = new FixtureRunner(c);
    const agent = new LayaAgent({ config: fixture.config, tokenizer, runner, batchSize: 64 });
    expect(await agent.predict(c.state, c.questions)).toEqual(c.result);
    expect(runner.calls.length).toBe(1);
  });

  it("splits large requests into batchSize chunks", async () => {
    const c = fixture.cases.find((x) => x.name === "many_questions")!;
    const runner = new FixtureRunner(c);
    const agent = new LayaAgent({ config: fixture.config, tokenizer, runner, batchSize: 16 });
    expect(await agent.predict(c.state, c.questions)).toEqual(c.result);
    expect(runner.calls.map((b) => b.rows)).toEqual([16, 4]);
  });

  it("rejects non-dictionary questions and empty requests", async () => {
    const agent = new LayaAgent({ config: fixture.config, tokenizer, runner: new FixtureRunner(fixture.cases[0]!) });
    await expect(agent.predict("x", [] as never)).rejects.toThrow("questions must be a dictionary");
    expect(await agent.predict("x", {})).toEqual({
      model: "laya-rl-agent",
      answers: {},
      usage: { input_tokens: 0, output_tokens: 0 },
    });
  });
});
```

Note: the `many_questions` chunk test replays rows 0–15 then 16–19 of the same padded fixture batch; the padded logits are position-independent so this is valid.

- [ ] **Step 2: Run to verify failure**

Run: `LAYA_TOKENIZER_DIR=<bundle>/tokenizer pnpm --filter @laya-mlx/web test -- agent`
Expected: FAIL, cannot find module `../src/agent.ts`

- [ ] **Step 3: Implement `agent.ts`**

```ts
import { formatAnswers } from "./calibration.ts";
import { buildSequence, collate } from "./prompt.ts";
import { toInternal } from "./questions.ts";
import type { LayaTokenizer } from "./tokenizer.ts";
import type {
  AgentConfig,
  Answer,
  Batch,
  InternalQuestion,
  PredictResult,
  PreparedItem,
  Question,
  RunnerOutput,
  State,
} from "./types.ts";

/** Executes one collated batch. `OnnxRunner` in session.ts is the production implementation. */
export interface Runner {
  run(batch: Batch): Promise<RunnerOutput>;
}

export interface LayaAgentOptions {
  config: AgentConfig;
  tokenizer: LayaTokenizer;
  runner: Runner;
  /** Questions per forward pass; Python defaults to 16. */
  batchSize?: number;
}

/** Port of Python `Agent` without model loading; see `LayaAgent.load` in session.ts. */
export class LayaAgent {
  readonly config: AgentConfig;
  readonly tokenizer: LayaTokenizer;
  readonly runner: Runner;
  readonly batchSize: number;

  constructor(options: LayaAgentOptions) {
    const batchSize = options.batchSize ?? 16;
    if (!Number.isInteger(batchSize) || batchSize < 1) throw new Error("batchSize must be a positive integer");
    this.config = options.config;
    this.tokenizer = options.tokenizer;
    this.runner = options.runner;
    this.batchSize = batchSize;
  }

  /** Python `Agent.prepare`: upstream-compatible token sequences for every question. */
  prepare(state: State, questions: Record<string, Question>): { items: PreparedItem[]; internal: InternalQuestion[] } {
    if (typeof questions !== "object" || questions === null || Array.isArray(questions)) {
      throw new Error("questions must be a dictionary keyed by question id");
    }
    const items: PreparedItem[] = [];
    const internal: InternalQuestion[] = [];
    for (const [qid, definition] of Object.entries(questions)) {
      const q = toInternal(definition);
      const item = buildSequence(this.tokenizer, state, q, this.config.max_len, this.config.head_max_len);
      if (item.markers.length !== renderedOptionCount(q)) {
        throw new Error(`Question ${JSON.stringify(qid)} has too many options for the token budget`);
      }
      items.push(item);
      internal.push(q);
    }
    return { items, internal };
  }

  /** Python `Agent.predict` / `system_one`. */
  async predict(state: State, questions: Record<string, Question>): Promise<PredictResult> {
    const { items, internal } = this.prepare(state, questions);
    const questionIds = Object.keys(questions);
    const answers: Record<string, Answer> = {};
    let inputTokens = 0;
    for (let start = 0; start < items.length; start += this.batchSize) {
      const chunk = items.slice(start, start + this.batchSize);
      const output = await this.runner.run(collate(chunk, this.tokenizer.padTokenId));
      const result = formatAnswers({
        config: this.config,
        questionIds: questionIds.slice(start, start + chunk.length),
        internal: internal.slice(start, start + chunk.length),
        items: chunk,
        logits: output.logits,
        actLogits: output.actLogits,
        markers: output.markers,
        actions: output.actions,
      });
      Object.assign(answers, result.answers);
      inputTokens += result.usage.input_tokens;
    }
    return { model: "laya-rl-agent", answers, usage: { input_tokens: inputTokens, output_tokens: 0 } };
  }
}

function renderedOptionCount(q: InternalQuestion): number {
  return q.t === "choice" ? Object.keys(q.crit).length : q.t === "score" ? q.crit.length : 2;
}
```

`src/index.ts`:

```ts
export { LayaAgent, type LayaAgentOptions, type Runner } from "./agent.ts";
export { confidenceFromProbs, formatAnswers, softmax, tempBucket } from "./calibration.ts";
export { buildPrefix, buildSequence, collate, serializeState } from "./prompt.ts";
export { pyJson } from "./pyjson.ts";
export { renderCriterion, renderOptions, toInternal } from "./questions.ts";
export { LayaTokenizer } from "./tokenizer.ts";
export * from "./types.ts";
```

- [ ] **Step 4: Run the tests**

Run: `LAYA_TOKENIZER_DIR=<bundle>/tokenizer pnpm --filter @laya-mlx/web test`
Expected: every test file passes (pyjson, questions, tokenizer, prompt, calibration, agent)

- [ ] **Step 5: Typecheck, format, commit**

```bash
pnpm typecheck && pnpm format:write
git add web/packages/laya-web/src web/packages/laya-web/test
git commit -m "Add LayaAgent with an injectable runner"
```

---

### Task 8: Bundle loading and the ONNX runner

**Files:**
- Create: `web/packages/laya-web/src/session.ts`
- Modify: `web/packages/laya-web/src/index.ts`

No Node test: this module needs `fetch` of a real bundle, the Cache API and onnxruntime-web. It is exercised by the Playwright parity test in Task 9.

- [ ] **Step 1: Implement `session.ts`**

```ts
import * as ort from "onnxruntime-web";

import { LayaAgent } from "./agent.ts";
import type { Runner } from "./agent.ts";
import { LayaTokenizer } from "./tokenizer.ts";
import type { AgentConfig, Batch, RunnerOutput } from "./types.ts";

export type Provider = "webgpu" | "wasm";

export interface LoadProgress {
  file: string;
  received: number;
  total: number | null;
}

export interface LoadOptions {
  /** Execution providers to try in order. Defaults to ["webgpu", "wasm"]. */
  providers?: Provider[];
  /** Cache API bucket for model.onnx; null disables caching. Defaults to "laya-models". */
  cacheName?: string | null;
  onProgress?: (progress: LoadProgress) => void;
  batchSize?: number;
  /** Where onnxruntime-web finds its .wasm/.mjs files; required when they are not next to the page. */
  wasmPaths?: string;
}

export interface OnnxConfig {
  format: "laya-onnx";
  format_version: number;
  dtype: "float32" | "float16";
  opset: number;
  inputs: string[];
  outputs: string[];
}

export interface Bundle {
  baseUrl: string;
  config: AgentConfig;
  onnxConfig: OnnxConfig;
  tokenizerJson: unknown;
  tokenizerConfig: unknown;
  model: ArrayBuffer;
}

const join = (base: string, name: string) => (base.endsWith("/") ? base : base + "/") + name;

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  return (await response.json()) as T;
}

/** Fetch a large file with progress; caches the bytes when a cache name is given and storage allows it. */
async function fetchBytes(
  url: string,
  cacheName: string | null,
  onProgress: LoadOptions["onProgress"],
): Promise<ArrayBuffer> {
  let cache: Cache | null = null;
  if (cacheName !== null && typeof caches !== "undefined") {
    try {
      cache = await caches.open(cacheName);
      const hit = await cache.match(url);
      if (hit) {
        const bytes = await hit.arrayBuffer();
        onProgress?.({ file: url, received: bytes.byteLength, total: bytes.byteLength });
        return bytes;
      }
    } catch {
      cache = null;
    }
  }
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || null;
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress?.({ file: url, received, total });
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  if (cache) {
    try {
      await cache.put(url, new Response(bytes, { headers: { "content-length": String(received) } }));
    } catch {
      // Quota exceeded or storage disabled: the model still runs, it is just fetched again next time.
    }
  }
  return bytes.buffer;
}

/** Download every file of a `laya-mlx export-onnx` bundle from a directory URL. */
export async function loadBundle(baseUrl: string, options: LoadOptions = {}): Promise<Bundle> {
  const cacheName = options.cacheName === undefined ? "laya-models" : options.cacheName;
  const [config, onnxConfig, tokenizerJson, tokenizerConfig] = await Promise.all([
    fetchJson<AgentConfig>(join(baseUrl, "rl_agent_config.json")),
    fetchJson<OnnxConfig>(join(baseUrl, "onnx_config.json")),
    fetchJson<unknown>(join(baseUrl, "tokenizer/tokenizer.json")),
    fetchJson<unknown>(join(baseUrl, "tokenizer/tokenizer_config.json")),
  ]);
  if (onnxConfig.format !== "laya-onnx") throw new Error(`Not a Laya ONNX bundle: ${baseUrl}`);
  const model = await fetchBytes(join(baseUrl, "model.onnx"), cacheName, options.onProgress);
  return { baseUrl, config, onnxConfig, tokenizerJson, tokenizerConfig, model };
}

/** Runs a Laya ONNX graph through onnxruntime-web. */
export class OnnxRunner implements Runner {
  private constructor(
    readonly session: ort.InferenceSession,
    readonly provider: Provider,
  ) {}

  static async create(model: ArrayBuffer, options: LoadOptions = {}): Promise<OnnxRunner> {
    if (options.wasmPaths) ort.env.wasm.wasmPaths = options.wasmPaths;
    const providers = options.providers ?? ["webgpu", "wasm"];
    let lastError: unknown;
    for (const provider of providers) {
      if (provider === "webgpu" && !("gpu" in navigator)) continue;
      try {
        const session = await ort.InferenceSession.create(model, {
          executionProviders: [provider],
          graphOptimizationLevel: "all",
        });
        return new OnnxRunner(session, provider);
      } catch (error) {
        lastError = error;
      }
    }
    throw new Error(`No execution provider could load the model: ${String(lastError)}`);
  }

  async run(batch: Batch): Promise<RunnerOutput> {
    const feeds = {
      input_ids: new ort.Tensor("int64", batch.inputIds, [batch.rows, batch.length]),
      attention_mask: new ort.Tensor("int64", batch.attentionMask, [batch.rows, batch.length]),
      marker_pos: new ort.Tensor("int64", batch.markerPos, [batch.rows, batch.markers]),
      marker_mask: new ort.Tensor("bool", batch.markerMask, [batch.rows, batch.markers]),
      qtype: new ort.Tensor("int64", batch.qtype, [batch.rows]),
    };
    const output = await this.session.run(feeds);
    const logits = output["logits"]!;
    const actLogits = output["act_logits"]!;
    return {
      rows: batch.rows,
      markers: Number(logits.dims[1]),
      actions: Number(actLogits.dims[1]),
      logits: logits.data as Float32Array,
      actLogits: actLogits.data as Float32Array,
    };
  }
}

export interface LoadedAgent {
  agent: LayaAgent;
  bundle: Bundle;
  provider: Provider;
}

/** One call from bundle URL to a ready agent; what the demos use. */
export async function loadAgent(baseUrl: string, options: LoadOptions = {}): Promise<LoadedAgent> {
  const bundle = await loadBundle(baseUrl, options);
  const runner = await OnnxRunner.create(bundle.model, options);
  const tokenizer = new LayaTokenizer(bundle.tokenizerJson as never, bundle.tokenizerConfig as never);
  const agent = new LayaAgent({ config: bundle.config, tokenizer, runner, ...(options.batchSize ? { batchSize: options.batchSize } : {}) });
  return { agent, bundle, provider: runner.provider };
}
```

Append to `src/index.ts`:

```ts
export { loadAgent, loadBundle, OnnxRunner } from "./session.ts";
export type { Bundle, LoadedAgent, LoadOptions, LoadProgress, OnnxConfig, Provider } from "./session.ts";
```

- [ ] **Step 2: Typecheck**

Run: `pnpm typecheck`
Expected: no errors. If `onnxruntime-web` types complain about `BigInt64Array` for int64 tensors, cast the data with `as unknown as BigInt64Array`; the runtime accepts it.

- [ ] **Step 3: Commit**

```bash
pnpm format:write
git add web/packages/laya-web/src/session.ts web/packages/laya-web/src/index.ts
git commit -m "Add bundle loading with Cache API and the onnxruntime-web runner"
```

---

### Task 9: Demo app skeleton with the browser parity page

**Files:**
- Create: `web/apps/demo/package.json`, `web/apps/demo/tsconfig.json`, `web/apps/demo/vite.config.ts`
- Create: `web/apps/demo/index.html`, `web/apps/demo/parity.html`, `web/apps/demo/src/parity.ts`, `web/apps/demo/src/model-url.ts`
- Create: `web/apps/demo/playwright.config.ts`, `web/apps/demo/e2e/parity.spec.ts`
- Create: `web/apps/demo/public/fixtures/parity-multilingual.json` (symlink to `../../../../fixtures/parity-multilingual.json`)

- [ ] **Step 1: Package files**

`web/apps/demo/package.json`:

```json
{
  "name": "laya-demo",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "typecheck": "tsc -p tsconfig.json",
    "test": "echo \"no unit tests in laya-demo yet\"",
    "test:browser": "playwright test"
  },
  "dependencies": {
    "@laya-mlx/web": "workspace:*",
    "onnxruntime-web": "^1.30.0"
  },
  "devDependencies": {
    "@playwright/test": "^1.63.0",
    "vite": "^8.0.0",
    "vite-plugin-static-copy": "^3.0.0"
  }
}
```

`web/apps/demo/tsconfig.json`:

```json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": { "types": ["vite/client"] },
  "include": ["src", "e2e", "vite.config.ts", "playwright.config.ts"]
}
```

`web/apps/demo/vite.config.ts`:

```ts
import { resolve } from "node:path";
import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

// onnxruntime-web loads its wasm/worker files by URL at runtime; ship them next to the pages.
export default defineConfig({
  base: process.env.VITE_BASE ?? "/",
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        index: resolve(__dirname, "index.html"),
        parity: resolve(__dirname, "parity.html"),
      },
    },
  },
  plugins: [
    viteStaticCopy({
      targets: [{ src: "node_modules/onnxruntime-web/dist/*.{wasm,mjs}", dest: "ort" }],
    }),
  ],
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
```

`web/apps/demo/src/model-url.ts`:

```ts
/** Directory URL of the ONNX bundle; override with VITE_LAYA_MODEL_URL (for example /models/laya-multilingual-onnx-fp16/). */
export const MODEL_URL: string =
  import.meta.env.VITE_LAYA_MODEL_URL ?? "https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/";

/** Where the copied onnxruntime-web assets live relative to the site root. */
export const ORT_WASM_PATHS: string = `${import.meta.env.BASE_URL}ort/`;
```

- [ ] **Step 2: Pages**

`web/apps/demo/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Laya in the browser</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
    <h1>Laya in the browser</h1>
    <p>Typed decisions from a Laya checkpoint, running in onnxruntime-web (WebGPU).</p>
    <ul>
      <li><a href="parity.html">Parity check</a> — runs the 63-question fixture through the model</li>
    </ul>
  </body>
</html>
```

`web/apps/demo/parity.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Laya parity</title>
  </head>
  <body>
    <h1>Parity: browser vs Python</h1>
    <p id="status">idle</p>
    <pre id="report"></pre>
    <script type="module" src="/src/parity.ts"></script>
  </body>
</html>
```

`web/apps/demo/src/parity.ts`:

```ts
import { loadAgent, softmax, tempBucket } from "@laya-mlx/web";
import type { PredictResult, Question, State } from "@laya-mlx/web";

import { MODEL_URL, ORT_WASM_PATHS } from "./model-url.ts";

interface FixtureCase {
  name: string;
  state: State;
  questions: Record<string, Question>;
  items: { markers: number[]; qtype: number }[];
  logits: number[][];
  result: PredictResult;
}
interface Fixture {
  config: { temperature: number[]; temperature_by_options: Record<string, number> };
  cases: FixtureCase[];
}

export interface ParityReport {
  provider: string;
  questions: number;
  argmax_agreements: number;
  probability_max_abs_error: number;
  public_result_equal: number;
  timings_ms: Record<string, number>;
}

const status = document.getElementById("status")!;
const report = document.getElementById("report")!;
const modelUrl = new URLSearchParams(location.search).get("model") ?? MODEL_URL;

async function main(): Promise<ParityReport> {
  const fixture: Fixture = await (await fetch(`${import.meta.env.BASE_URL}fixtures/parity-multilingual.json`)).json();
  status.textContent = `loading ${modelUrl}`;
  const { agent, provider } = await loadAgent(modelUrl, {
    wasmPaths: ORT_WASM_PATHS,
    batchSize: 64,
    onProgress: (p) => {
      status.textContent = `${p.file}: ${(p.received / 1e6).toFixed(0)} / ${p.total ? (p.total / 1e6).toFixed(0) : "?"} MB`;
    },
  });
  status.textContent = `running on ${provider}`;
  const summary: ParityReport = {
    provider,
    questions: 0,
    argmax_agreements: 0,
    probability_max_abs_error: 0,
    public_result_equal: 0,
    timings_ms: {},
  };
  for (const c of fixture.cases) {
    const started = performance.now();
    const result = await agent.predict(c.state, c.questions);
    summary.timings_ms[c.name] = Math.round(performance.now() - started);
    const ids = Object.keys(c.questions);
    ids.forEach((qid, row) => {
      const k = c.items[row]!.markers.length;
      const qtype = c.items[row]!.qtype;
      const scale =
        fixture.config.temperature_by_options[tempBucket(qtype, k)] ?? fixture.config.temperature[qtype]!;
      const expected = softmax(c.logits[row]!.slice(0, k).map((v) => v / Math.max(1e-3, scale)));
      const answer = result.answers[qid]!;
      const actual =
        answer.type === "noul" ? [1 - answer.noul, answer.noul] : Object.values(answer.probabilities);
      summary.questions += 1;
      if (expected.indexOf(Math.max(...expected)) === actual.indexOf(Math.max(...actual))) summary.argmax_agreements += 1;
      expected.forEach((v, i) => {
        summary.probability_max_abs_error = Math.max(summary.probability_max_abs_error, Math.abs(v - actual[i]!));
      });
    });
    if (JSON.stringify(result) === JSON.stringify(c.result)) summary.public_result_equal += 1;
  }
  status.textContent = `done: ${summary.argmax_agreements}/${summary.questions} argmax on ${provider}`;
  report.textContent = JSON.stringify(summary, null, 2);
  return summary;
}

main()
  .then((summary) => console.log("[laya] RESULT " + JSON.stringify(summary)))
  .catch((error) => {
    status.textContent = `ERROR ${String(error)}`;
    console.error(error);
  });
```

Create the fixture symlink so Vite serves it from `public/`:

```bash
mkdir -p web/apps/demo/public/fixtures
ln -s ../../../../fixtures/parity-multilingual.json web/apps/demo/public/fixtures/parity-multilingual.json
```

- [ ] **Step 3: Playwright config and env-gated spec**

`web/apps/demo/playwright.config.ts`:

```ts
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 15 * 60 * 1000,
  use: { baseURL: "http://127.0.0.1:4173" },
  webServer: { command: "pnpm vite --port 4173 --strictPort", url: "http://127.0.0.1:4173", reuseExistingServer: true },
  projects: [
    {
      name: "chromium-webgpu",
      use: {
        browserName: "chromium",
        launchOptions: { args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--ignore-gpu-blocklist"] },
      },
    },
  ],
});
```

`web/apps/demo/e2e/parity.spec.ts`:

```ts
import { expect, test } from "@playwright/test";

const modelUrl = process.env.LAYA_MODEL_URL;

test.skip(!modelUrl, "set LAYA_MODEL_URL=<bundle directory URL> to run the real-model parity check");

test("browser predictions match the Python fixtures", async ({ page }) => {
  const lines: string[] = [];
  page.on("console", (message) => lines.push(message.text()));
  await page.goto(`/parity.html?model=${encodeURIComponent(modelUrl!)}`);
  await expect(page.locator("#status")).toContainText(/done:|ERROR/, { timeout: 14 * 60 * 1000 });
  const result = lines.find((line) => line.startsWith("[laya] RESULT "));
  expect(result, lines.join("\n")).toBeDefined();
  const summary = JSON.parse(result!.slice("[laya] RESULT ".length));
  expect(summary.provider).toBe("webgpu");
  expect(summary.argmax_agreements).toBe(summary.questions);
  expect(summary.questions).toBe(63);
  expect(summary.probability_max_abs_error).toBeLessThan(2e-2);
});
```

- [ ] **Step 4: Install, typecheck, run the dev server once by hand**

```bash
cd /Users/mz/ghq/github.com/mizorewww/laya-mlx/web && pnpm install && pnpm typecheck
pnpm --filter laya-demo exec playwright install chromium
```

Serve the local bundle for the test: copy or symlink the float16 export into `web/apps/demo/public/models/laya-multilingual-onnx-fp16` (the `models/` pattern is git-ignored at the repository root; add `web/apps/demo/public/models/` to `.gitignore` too).

```bash
mkdir -p web/apps/demo/public/models
ln -s <absolute path to onnx-multilingual-float16> web/apps/demo/public/models/laya-multilingual-onnx-fp16
```

- [ ] **Step 5: Run the browser parity test**

Run: `cd web && LAYA_MODEL_URL=http://127.0.0.1:4173/models/laya-multilingual-onnx-fp16/ pnpm test:browser`
Expected: 1 passed; the console line shows `argmax_agreements: 63`, `probability_max_abs_error` around 1.2e-2 and `public_result_equal` below 16 (float16 changes fourth decimals; that is expected and not asserted).

If the page reports `no available backend found`, the ORT assets are not being served: check `http://127.0.0.1:4173/ort/ort-wasm-simd-threaded.jsep.mjs` in a browser and the `wasmPaths` value.

- [ ] **Step 6: Commit**

```bash
pnpm format:write
git add web/apps/demo web/pnpm-lock.yaml .gitignore
git commit -m "Add the demo site skeleton with a browser parity page"
```

---

### Task 10: Publish the float16 bundle to Hugging Face

**Files:**
- Create: `<bundle dir>/README.md` (model card, not in git; its text is also saved as `docs/onnx-model-card.md`)
- Create: `docs/onnx-model-card.md`

- [ ] **Step 1: Write the model card**

`docs/onnx-model-card.md` (copy the same text to `<bundle dir>/README.md`):

```markdown
---
license: apache-2.0
base_model: convaiinnovations/laya-multilingual
library_name: onnxruntime
tags:
  - onnx
  - onnxruntime-web
  - webgpu
  - typed-decisions
  - laya
---

# Laya multilingual — ONNX (float16)

ONNX export of [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)
(mmBERT-base encoder, 322M parameters) for onnxruntime, including onnxruntime-web with the WebGPU backend.
Produced by [laya-mlx](https://github.com/mizorewww/laya-mlx) `laya-mlx export-onnx` from the published
[aac6fef/laya-multilingual-mlx](https://huggingface.co/aac6fef/laya-multilingual-mlx) checkpoint
(source revision `052592a15d198d9ad47da779604259b10b47b7aa`).

| File | Purpose |
|---|---|
| `model.onnx` | Encoder, decision head, scorer and action head. Opset 18, standard operators only, float16 weights with a float32 decision tail. |
| `rl_agent_config.json` | Prompt limits and calibration temperatures. |
| `tokenizer/` | Hugging Face tokenizer files. |
| `onnx_config.json` | Export metadata: dtype, opset, input and output names. |

Inputs: `input_ids`, `attention_mask` (int64 `[batch, sequence]`), `marker_pos` (int64 `[batch, markers]`),
`marker_mask` (bool `[batch, markers]`), `qtype` (int64 `[batch]`). Outputs: `logits` `[batch, markers]`,
`act_logits` `[batch, actions]`, both float32. Prompt construction and calibration follow upstream Laya;
the reference implementation is `laya_mlx.common` and the browser port `@laya-mlx/web` in the repository.

Validation on the 63-question parity fixtures against the MLX runtime: 63/63 selected answers agree,
maximum probability error 5.1e-4 (onnxruntime CPU), 1.2e-2 (onnxruntime-web WebGPU float16).
See `docs/ONNX_EXPORT.md` in the repository for the method and the browser measurements.

This is an independent port, not an official Convai Innovations release. Weights are Apache-2.0 as upstream.
```

- [ ] **Step 2: Upload**

```bash
cd /Users/mz/ghq/github.com/mizorewww/laya-mlx
cp docs/onnx-model-card.md <bundle dir>/README.md
uv run hf upload mizchi/laya-multilingual-onnx <bundle dir> . --repo-type model --commit-message "Add float16 ONNX export of laya-multilingual"
```

Expected: the command prints the repo URL `https://huggingface.co/mizchi/laya-multilingual-onnx`. The 647 MB upload takes a few minutes.

- [ ] **Step 3: Verify CORS and the directory layout from the command line**

```bash
curl -sI -L https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/model.onnx | grep -iE "^(HTTP|access-control-allow-origin|content-length)"
curl -s https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/onnx_config.json
```

Expected: a `200` after redirects, `access-control-allow-origin: *`, `content-length: 6468…`, and the JSON with `"format": "laya-onnx"`.

- [ ] **Step 4: Run the browser parity test against Hugging Face**

Run: `cd web && LAYA_MODEL_URL=https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/ pnpm test:browser`
Expected: 1 passed (the first run downloads 647 MB).

- [ ] **Step 5: Commit the card**

```bash
git add docs/onnx-model-card.md
git commit -m "Add the Hugging Face model card for the ONNX bundle"
```

---

### Task 11: CI job and documentation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `web/README.md`
- Modify: `docs/ONNX_EXPORT.md` (replace the "browser harness is not yet part of this repository" paragraph)

- [ ] **Step 1: Add the `web` job**

Append to `.github/workflows/ci.yml` under `jobs:`:

```yaml
  web:
    runs-on: ubuntu-latest
    timeout-minutes: 15
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
      - run: pnpm format
      - run: pnpm typecheck
      - run: pnpm test
      - run: pnpm build
```

The Node tests that need `LAYA_TOKENIZER_DIR` skip themselves in CI; `pyjson`, `questions` and the `serializeState`/helper tests still run.

- [ ] **Step 2: Write `web/README.md`**

```markdown
# Laya in the browser

`packages/laya-web` (`@laya-mlx/web`) runs a Laya ONNX bundle with onnxruntime-web and returns the same
JSON as the Python `Agent.predict`. `apps/demo` is the static site (parity page now; Snake and Chess demos
follow).

```bash
pnpm install
pnpm typecheck && pnpm test                      # Node tests (fixture based, no model needed)
LAYA_TOKENIZER_DIR=<bundle>/tokenizer pnpm test  # also the tokenizer/prompt/agent parity tests
pnpm --filter laya-demo dev                      # http://localhost:5173
LAYA_MODEL_URL=https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/ pnpm test:browser
```

`python -m benchmarks.web_fixtures` regenerates `fixtures/parity-multilingual.json` from the MLX runtime.
The demo reads the model from `VITE_LAYA_MODEL_URL` (default: the Hugging Face bundle above); point it at
`/models/<bundle>/` under `apps/demo/public/` for offline development.

Usage:

```ts
import { loadAgent } from "@laya-mlx/web";

const { agent, provider } = await loadAgent("https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/main/", {
  wasmPaths: "/ort/",
  onProgress: (p) => console.log(p.received, p.total),
});
const result = await agent.predict("I was billed twice.", {
  department: { type: "choice", instructions: "Who should handle this?", criteria: ["billing", "sales"] },
});
console.log(provider, result.answers.department);
```
```

- [ ] **Step 3: Update `docs/ONNX_EXPORT.md`**

Replace the paragraph starting "The browser harness (fixture dump plus a static page and Playwright runner) is not yet part of this repository." with:

```markdown
The browser runtime lives in `web/packages/laya-web` and the measurement page is `web/apps/demo/parity.html`;
`pnpm test:browser` with `LAYA_MODEL_URL` set repeats the parity check above. The published bundle is
[mizchi/laya-multilingual-onnx](https://huggingface.co/mizchi/laya-multilingual-onnx).
```

- [ ] **Step 4: Run everything once more**

```bash
cd /Users/mz/ghq/github.com/mizorewww/laya-mlx
uv run ruff check . && uv run ruff format --check . && LAYA_MLX_TEST_DEVICE=cpu uv run pytest -q
cd web && pnpm format && pnpm typecheck && LAYA_TOKENIZER_DIR=<bundle>/tokenizer pnpm test && pnpm build
```

Expected: Python suite green; web format/typecheck/test/build green; `apps/demo/dist/ort/` contains the ORT wasm files.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml web/README.md docs/ONNX_EXPORT.md
git commit -m "Add the web CI job and document the browser runtime"
```

---

## Self-review

- Spec coverage: types/prompt/calibration/session/agent (§1) → Tasks 3–8; tokenizer choice and its test → Task 4; Node tests 1–2 → Tasks 5–7; Playwright parity 3 → Task 9; bundle URL convention and cache → Task 8; Hugging Face upload with model card, `VITE_LAYA_MODEL_URL`, ORT assets in dist, CI job (§4) → Tasks 9–11. `pages.yml` deployment is deliberately left to Plan 2 (it needs the Snake page to be worth deploying).
- Placeholders: `<bundle dir>` / `<bundle>` are the local export directory the engineer chooses (Task 4 Step 1 names the existing one); everything else is literal.
- Type consistency: `PreparedItem`, `Batch`, `RunnerOutput`, `InternalQuestion` are defined in Task 3 and used with the same field names in Tasks 5–9; `Runner` is defined in Task 7 and implemented by `OnnxRunner` in Task 8; `formatAnswers` takes `FormatInput` in Task 6 and is called with the same fields in Task 7.
