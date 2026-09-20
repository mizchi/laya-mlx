/**
 * Policy layer: prompt construction, model-agnostic decision making and the
 * deterministic cycle safety shield. Faithful TypeScript port of
 * ../../../laya_mlx/snake/policy.py's `build_prompt` and `LayaPolicy.decide`.
 */
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

function findPreferred(safe: MoveInfo[]): Direction | "NONE" {
  if (safe.length === 0) return "NONE";
  let best = safe[0]!;
  for (const move of safe) {
    if (move.advance > best.advance) best = move;
  }
  return best.direction;
}

export function buildPrompt(
  game: SnakeGame,
  moves: MoveInfo[],
  reachable: boolean,
  space: number,
  kind: PromptKind,
): Prompt {
  const safe = moves.filter((m) => m.safe);
  const preferred = findPreferred(safe);

  if (kind === "compact") {
    const state = `Safe route: ${safe.length ? "yes" : "no"}. Food reachable through empty cells: ${reachable ? "yes" : "no"}.`;
    const criteria: Record<string, string> = {};
    for (const move of moves) {
      criteria[move.direction] = !move.legal
        ? "Blocked. Collision."
        : !move.safe
          ? "Unsafe. Traps the snake."
          : move.eats
            ? "Safe. Eat food now. Best."
            : move.direction === preferred
              ? "Safe. Best route to food."
              : "Safe. Slower route.";
    }
    const questions: Record<string, Question> = {
      move: {
        type: "choice",
        instructions: "Choose the best safe move toward food.",
        criteria,
      },
      risk: {
        type: "noul",
        instructions: "Is a safe route available?",
      },
      food: {
        type: "noul",
        instructions: "Is food reachable through empty cells?",
      },
    };
    return { state, questions, preferred };
  }

  const descriptions: Record<string, string> = {};
  for (const move of moves) {
    descriptions[move.direction] = !move.legal
      ? `Collision: ${move.reason}. Unsafe.`
      : !move.safe
        ? "Unsafe route. Risk of trapping the snake."
        : move.eats
          ? "Safe. Eat the food immediately. Best move."
          : move.direction === preferred
            ? "Safe. Best progress toward food."
            : "Safe but less progress toward food.";
  }
  const state =
    `Snake game. ${safe.length} safe directions available. ` +
    `Food reachable through empty cells: ${reachable ? "yes" : "no"}. ` +
    `Open cells: ${space}. Snake length: ${game.body.length}. ` +
    `${safe.length ? "There is a safe route forward." : "The snake is trapped."}`;
  const questions: Record<string, Question> = {
    move: {
      type: "choice",
      instructions: "Select the safest move with best progress toward food. Avoid collisions.",
      criteria: descriptions,
    },
    risk: {
      type: "noul",
      instructions: "Is there a safe route forward for the snake?",
    },
    food: {
      type: "noul",
      instructions: "Is food reachable through the currently empty cells?",
    },
  };
  return { state, questions, preferred };
}

export interface ShieldResult {
  proposed: Direction;
  executed: Direction;
  intervened: boolean;
}

function argmaxBy<T extends string>(keys: readonly T[], value: (key: T) => number): T {
  let best = keys[0]!;
  let bestValue = value(best);
  for (const key of keys) {
    const v = value(key);
    if (v > bestValue) {
      bestValue = v;
      best = key;
    }
  }
  return best;
}

export function applyShield(
  probabilities: Record<Direction, number>,
  safeDirections: Direction[],
  guarded: boolean,
): ShieldResult {
  const proposed = argmaxBy(DIRECTIONS, (d) => probabilities[d]);
  const executed =
    guarded && !safeDirections.includes(proposed)
      ? argmaxBy(safeDirections, (d) => probabilities[d])
      : proposed;
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

function isValidProbability(value: number): boolean {
  return Number.isFinite(value) && value >= 0 && value <= 1;
}

export async function decide(
  agent: SnakeAgent,
  game: SnakeGame,
  options: PolicyOptions,
): Promise<Decision> {
  const started = performance.now();
  const moves = game.moves();
  const safe = moves.filter((m) => m.safe);
  if (safe.length === 0 && options.guarded) {
    throw new Error("Cycle safety invariant violated: no safe action");
  }
  const [reachable, space] = game.foodReachability();
  const prompt = buildPrompt(game, moves, reachable, space, options.prompt);

  const inferenceStart = performance.now();
  const output = await agent.predict(prompt.state, prompt.questions);
  const inferenceMs = performance.now() - inferenceStart;

  const moveAnswer = output.answers.move;
  if (!moveAnswer || moveAnswer.type !== "choice") {
    throw new Error("Model returned an unexpected shape for the move answer");
  }
  const riskAnswer = output.answers.risk;
  if (!riskAnswer || riskAnswer.type !== "noul") {
    throw new Error("Model returned an unexpected shape for the risk answer");
  }
  const foodAnswer = output.answers.food;
  if (!foodAnswer || foodAnswer.type !== "noul") {
    throw new Error("Model returned an unexpected shape for the food answer");
  }

  const probabilities = {} as Record<Direction, number>;
  for (const direction of DIRECTIONS) {
    probabilities[direction] = moveAnswer.probabilities[direction] ?? Number.NaN;
  }
  const scores = [...DIRECTIONS.map((d) => probabilities[d]), riskAnswer.noul, foodAnswer.noul];
  if (scores.some((value) => !isValidProbability(value))) {
    throw new Error("Model returned an invalid probability; no move executed");
  }

  const allowed = safe.map((m) => m.direction);
  const shield = applyShield(probabilities, allowed, options.guarded);

  return {
    probabilities,
    proposed: shield.proposed,
    executed: shield.executed,
    safeDirections: allowed,
    intervened: shield.intervened,
    deadEndRisk: 1 - riskAnswer.noul,
    foodReachable: foodAnswer.noul,
    inferenceMs,
    decisionMs: performance.now() - started,
    inputTokens: output.usage.input_tokens,
    outputTokens: output.usage.output_tokens ?? 0,
    safeCount: safe.length,
    plannerBest: prompt.preferred,
  };
}
