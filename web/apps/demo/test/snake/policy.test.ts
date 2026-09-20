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
    expect(applyShield(p, ["UP", "DOWN"], true)).toEqual({
      proposed: "UP",
      executed: "UP",
      intervened: false,
    });
  });
  it("redirects to the most likely safe direction when guarded", () => {
    expect(applyShield(p, ["DOWN", "LEFT"], true)).toEqual({
      proposed: "UP",
      executed: "DOWN",
      intervened: true,
    });
  });
  it("executes the raw proposal when unguarded", () => {
    expect(applyShield(p, ["DOWN"], false)).toEqual({
      proposed: "UP",
      executed: "UP",
      intervened: false,
    });
  });
});

describe("decide", () => {
  it("returns a Decision with model probabilities, estimates and timings", async () => {
    const game = new SnakeGame(8, 6, 3, 4);
    const decision = await decide(new StubAgent(), game, { guarded: true, prompt: "compact" });
    expect(Object.keys(decision.probabilities)).toEqual(["UP", "DOWN", "LEFT", "RIGHT"]);
    expect(
      game
        .moves()
        .filter((m) => m.safe)
        .map((m) => m.direction),
    ).toContain(decision.executed);
    expect(decision.deadEndRisk).toBeGreaterThanOrEqual(0);
    expect(decision.foodReachable).toBeLessThanOrEqual(1);
    expect(decision.inputTokens).toBeGreaterThan(0);
    expect(decision.outputTokens).toBe(0);
    expect(decision.inferenceMs).toBeGreaterThanOrEqual(0);
  });
  it("throws when the shield invariant is violated", async () => {
    const game = new SnakeGame(8, 6, 3, 4);
    game.alive = false;
    await expect(
      decide(new StubAgent(), game, { guarded: true, prompt: "compact" }),
    ).rejects.toThrow(/no safe action/);
  });
  it("rejects invalid probabilities from the model", async () => {
    const game = new SnakeGame(8, 6, 3, 4);
    const bad = new StubAgent({ probability: () => Number.NaN });
    await expect(decide(bad, game, { guarded: true, prompt: "compact" })).rejects.toThrow(
      /invalid probability/,
    );
  });
});
