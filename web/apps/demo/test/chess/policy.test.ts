import { describe, expect, it } from "vitest";

import type { Candidate } from "../../src/chess/candidates.ts";
import { ChessGame } from "../../src/chess/game.ts";
import { applyShield, buildPrompt, decide } from "../../src/chess/policy.ts";
import { StubAgent } from "../../src/stub-agent.ts";

const cand = (san: string, score: number, best = false, allowsMate = false): Candidate => ({
  san,
  from: "a1",
  to: "a2",
  score,
  description: best ? "Quiet move. Safe. Best." : "Quiet move. Safe.",
  best,
  allowsMate,
});

describe("buildPrompt", () => {
  it("describes the position and lists candidates as choice criteria", () => {
    const game = new ChessGame();
    const prompt = buildPrompt(game, [cand("e4", 0.2, true), cand("d4", 0.2)]);
    expect(prompt.state).toBe("Chess. We play white. Move 1. Material: even. Not in check.");
    expect(prompt.questions["move"]).toEqual({
      type: "choice",
      instructions: "Choose the strongest safe move.",
      criteria: { e4: "Quiet move. Safe. Best.", d4: "Quiet move. Safe." },
    });
    expect(prompt.questions["danger"]).toEqual({
      type: "noul",
      instructions: "Is our king in danger?",
    });
    expect(Object.keys(prompt.questions)).toEqual(["move", "danger"]);
  });
  it("reports material and check from our perspective", () => {
    const up = new ChessGame("k7/8/8/8/8/8/8/K6Q w - - 0 1");
    expect(buildPrompt(up, [cand("Qh8+", 9, true)]).state).toContain("Material: +9.");
    const down = new ChessGame("k7/8/8/8/8/8/8/K6Q b - - 0 1");
    expect(buildPrompt(down, [cand("Kb8", -9, true)]).state).toContain(
      "We play black. Move 1. Material: -9.",
    );
    const inCheck = new ChessGame("k7/8/8/8/8/8/8/K6q w - - 0 1");
    expect(buildPrompt(inCheck, [cand("Ka2", -9, true)]).state).toContain("In check.");
  });
});

describe("applyShield", () => {
  const c = [
    cand("Best", 2, true),
    cand("Ok", 0),
    cand("Blunder", -4),
    cand("Mate", -1000, false, true),
  ];
  const p = { Best: 0.1, Ok: 0.2, Blunder: 0.6, Mate: 0.1 };
  it("keeps a proposal within three pawns of the best", () => {
    expect(applyShield({ ...p, Ok: 0.9 }, c, true)).toEqual({
      proposed: "Ok",
      executed: "Ok",
      intervened: false,
    });
  });
  it("overrides a proposal more than three pawns worse than the best", () => {
    expect(applyShield(p, c, true)).toEqual({
      proposed: "Blunder",
      executed: "Best",
      intervened: true,
    });
  });
  it("overrides a proposal that allows mate when another candidate does not", () => {
    expect(applyShield({ ...p, Blunder: 0, Mate: 0.9 }, c, true)).toEqual({
      proposed: "Mate",
      executed: "Best",
      intervened: true,
    });
  });
  it("does nothing when unguarded", () => {
    expect(applyShield(p, c, false)).toEqual({
      proposed: "Blunder",
      executed: "Blunder",
      intervened: false,
    });
  });
  it("picks the first candidate on probability ties", () => {
    expect(applyShield({ Best: 0.25, Ok: 0.25, Blunder: 0.25, Mate: 0.25 }, c, true).proposed).toBe(
      "Best",
    );
  });
});

describe("decide", () => {
  it("returns a Decision whose executed move is a legal candidate", async () => {
    const game = new ChessGame();
    const d = await decide(new StubAgent(), game, { guarded: true });
    expect(d.candidates.map((c) => c.san)).toContain(d.executed);
    expect(Object.keys(d.probabilities)).toEqual(d.candidates.map((c) => c.san));
    expect(d.kingInDanger).toBeGreaterThanOrEqual(0);
    expect(d.kingInDanger).toBeLessThanOrEqual(1);
    expect(d.outputTokens).toBe(0);
    expect(d.inputTokens).toBeGreaterThan(0);
    expect(d.inferenceMs).toBeGreaterThanOrEqual(0);
    expect(d.decisionMs).toBeGreaterThanOrEqual(d.inferenceMs);
    expect(game.history).toEqual([]); // decide never mutates the game
  });
  it("throws on a finished game", async () => {
    const game = new ChessGame("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    await expect(decide(new StubAgent(), game, { guarded: true })).rejects.toThrow(
      /no legal moves/,
    );
  });
  it("rejects malformed and invalid answers", async () => {
    const game = new ChessGame();
    const shaped = (answers: unknown) =>
      ({
        predict: async () => ({
          model: "laya-rl-agent",
          answers,
          usage: { input_tokens: 1, output_tokens: 0 },
        }),
      }) as never;
    await expect(decide(shaped({}), game, { guarded: true })).rejects.toThrow(/move answer/);
    await expect(
      decide(
        shaped({
          move: {
            type: "choice",
            confidence: 1,
            action: { act_probability: 1 },
            choice: "e4",
            probabilities: { e4: 1 },
          },
        }),
        game,
        { guarded: true },
      ),
    ).rejects.toThrow(/danger answer/);
    const bad = new StubAgent({ probability: () => Number.NaN });
    await expect(decide(bad, game, { guarded: true })).rejects.toThrow(/invalid probability/);
  });
});
