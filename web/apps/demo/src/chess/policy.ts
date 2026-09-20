/**
 * Policy layer for chess: prompt construction, model-agnostic decision making
 * and the deterministic blunder shield. Sibling to ../snake/policy.ts — same
 * shape (Agent interface, buildPrompt, applyShield, decide), adapted to
 * chess.js candidates instead of Snake directions.
 */
import type { PredictResult, Question, State } from "@laya-mlx/web";

import { argmax } from "../argmax.ts";
import { type Candidate, evaluateMaterial, shortlist } from "./candidates.ts";
import type { ChessGame } from "./game.ts";

export interface ChessAgent {
  predict(state: State, questions: Record<string, Question>): Promise<PredictResult>;
}

export interface Prompt {
  state: string;
  questions: Record<string, Question>;
}

function materialText(material: number): string {
  if (material === 0) return "even";
  return material > 0 ? `+${material}` : `${material}`;
}

export function buildPrompt(game: ChessGame, candidates: Candidate[]): Prompt {
  const material = evaluateMaterial(game.chess);
  const moveNumber = game.chess.moveNumber();
  const color = game.turn === "w" ? "white" : "black";
  const check = game.chess.inCheck() ? "In check." : "Not in check.";
  const state = `Chess. We play ${color}. Move ${moveNumber}. Material: ${materialText(material)}. ${check}`;

  const criteria: Record<string, string> = {};
  for (const candidate of candidates) {
    criteria[candidate.san] = candidate.description;
  }

  const questions: Record<string, Question> = {
    move: {
      type: "choice",
      instructions: "Choose the strongest safe move.",
      criteria,
    },
    danger: {
      type: "noul",
      instructions: "Is our king in danger?",
    },
  };

  return { state, questions };
}

/**
 * Material deficit (in pawns) below the best candidate's score at which the
 * shield overrides the model's proposal. 3 is a minor piece down — the point
 * a beginner notices — so the shield steps in before the model gives one away.
 */
export const SHIELD_MARGIN = 3;

export interface ShieldResult {
  proposed: string;
  executed: string;
  intervened: boolean;
}

function probabilityFor(probabilities: Record<string, number>, san: string): number {
  const value = probabilities[san];
  if (value === undefined) {
    throw new Error(`applyShield: missing probability for ${san}`);
  }
  return value;
}

export function applyShield(
  probabilities: Record<string, number>,
  candidates: Candidate[],
  guarded: boolean,
): ShieldResult {
  const proposed = argmax(candidates, (c) => probabilityFor(probabilities, c.san)).san;
  if (!guarded) {
    return { proposed, executed: proposed, intervened: false };
  }

  const best = candidates.find((c) => c.best) ?? candidates[0]!;
  const chosen = candidates.find((c) => c.san === proposed) ?? best;

  const override =
    best.score - chosen.score >= SHIELD_MARGIN || (chosen.allowsMate && !best.allowsMate);
  const executed = override ? best.san : proposed;

  return { proposed, executed, intervened: executed !== proposed };
}

export interface Decision {
  probabilities: Record<string, number>;
  proposed: string;
  executed: string;
  intervened: boolean;
  kingInDanger: number;
  inferenceMs: number;
  decisionMs: number;
  inputTokens: number;
  outputTokens: number;
  candidates: Candidate[];
}

export interface PolicyOptions {
  guarded: boolean;
}

function isValidProbability(value: number): boolean {
  return Number.isFinite(value) && value >= 0 && value <= 1;
}

export async function decide(
  agent: ChessAgent,
  game: ChessGame,
  options: PolicyOptions,
): Promise<Decision> {
  const started = performance.now();
  const candidates = shortlist(game.chess);
  if (candidates.length === 0) {
    throw new Error("Chess: no legal moves");
  }

  const prompt = buildPrompt(game, candidates);

  const inferenceStart = performance.now();
  const output = await agent.predict(prompt.state, prompt.questions);
  const inferenceMs = performance.now() - inferenceStart;

  const moveAnswer = output.answers.move;
  if (!moveAnswer || moveAnswer.type !== "choice") {
    throw new Error("Model returned an unexpected shape for the move answer");
  }
  const dangerAnswer = output.answers.danger;
  if (!dangerAnswer || dangerAnswer.type !== "noul") {
    throw new Error("Model returned an unexpected shape for the danger answer");
  }

  const probabilities: Record<string, number> = {};
  for (const candidate of candidates) {
    probabilities[candidate.san] = moveAnswer.probabilities[candidate.san] ?? Number.NaN;
  }
  const scores = [...candidates.map((c) => probabilities[c.san]!), dangerAnswer.noul];
  if (scores.some((value) => !isValidProbability(value))) {
    throw new Error("Model returned an invalid probability; no move executed");
  }

  const shield = applyShield(probabilities, candidates, options.guarded);

  return {
    probabilities,
    proposed: shield.proposed,
    executed: shield.executed,
    intervened: shield.intervened,
    kingInDanger: dangerAnswer.noul,
    inferenceMs,
    decisionMs: performance.now() - started,
    inputTokens: output.usage.input_tokens,
    outputTokens: output.usage.output_tokens ?? 0,
    candidates,
  };
}
