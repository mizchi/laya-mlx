import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import type {
  AgentConfig,
  InternalQuestion,
  PredictResult,
  Question,
  State,
} from "../src/types.ts";

export interface FixtureCase {
  name: string;
  state: State;
  questions: Record<string, Question>;
  items: { ids: number[]; markers: number[]; qtype: number }[];
  /** Python `Agent._to_internal(q)` per question, in `questions` iteration order. */
  internal: InternalQuestion[];
  /** Python `render_options(internal[i])` per question. */
  options: string[][];
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
  revision: string;
  packages: Record<string, string>;
  config: AgentConfig;
  special_tokens: { cls: number; sep: number; pad: number; mask: number; mask_token: string };
  tokenizer_cases: { text: string; ids: number[] }[];
  cases: FixtureCase[];
}

const path = fileURLToPath(new URL("../../../fixtures/parity-multilingual.json", import.meta.url));
export const fixture: Fixture = JSON.parse(readFileSync(path, "utf8"));
