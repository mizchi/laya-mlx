import { describe, expect, it, vi } from "vitest";

import { GameLoop } from "../../src/snake/loop.ts";
import type { SnakeAgent } from "../../src/snake/policy.ts";
import { StubAgent } from "../../src/snake/stub-agent.ts";

const settings = {
  width: 8,
  height: 6,
  seed: 3,
  initialLength: 4,
  fps: 12,
  guarded: true,
  prompt: "compact" as const,
};

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
    const unsafe = new StubAgent({
      probability: (_, desc) =>
        desc.startsWith("Blocked") || desc.startsWith("Unsafe") ? 1 : 0.01,
    });
    const loop = new GameLoop(unsafe, settings);
    await loop.tick();
    expect(loop.lastDecision!.intervened).toBe(true);
    expect(loop.stats.interventions).toBe(1);
  });
  it("never overlaps decisions when tick is called concurrently", async () => {
    const slow = new StubAgent();
    const spy = vi.spyOn(slow, "predict");
    const loop = new GameLoop(slow, settings);
    await Promise.all([loop.tick(), loop.tick(), loop.tick()]);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(loop.game.ticks).toBe(1);
  });
  it("computes the paced delay and zero in max-speed mode", () => {
    const loop = new GameLoop(new StubAgent(), { ...settings, fps: 10 });
    const startedAt = performance.now();
    const delay = loop.delayAfter(startedAt);
    expect(delay).toBeGreaterThan(0);
    expect(delay).toBeLessThanOrEqual(100);
    loop.maxSpeed = true;
    expect(loop.delayAfter(startedAt)).toBe(0);
  });
  it("propagates a predict() rejection without mutating game state, then recovers", async () => {
    let fail = true;
    const agent: SnakeAgent = {
      predict: (state, questions) =>
        fail ? Promise.reject(new Error("boom")) : new StubAgent().predict(state, questions),
    };
    const loop = new GameLoop(agent, settings);
    const ticksBefore = loop.game.ticks;
    await expect(loop.tick()).rejects.toThrow("boom");
    expect(loop.stats.decisions).toBe(0);
    expect(loop.game.ticks).toBe(ticksBefore);
    fail = false;
    await loop.tick();
    expect(loop.stats.decisions).toBe(1);
    expect(loop.game.ticks).toBe(ticksBefore + 1);
  });
  it("rejects every concurrent tick() call when the shared decision rejects", async () => {
    const agent: SnakeAgent = { predict: () => Promise.reject(new Error("boom")) };
    const loop = new GameLoop(agent, settings);
    const results = await Promise.allSettled([loop.tick(), loop.tick(), loop.tick()]);
    expect(results.map((r) => r.status)).toEqual(["rejected", "rejected", "rejected"]);
  });
  it("counts one decision per second immediately after a single tick", async () => {
    const loop = new GameLoop(new StubAgent(), settings);
    await loop.tick();
    expect(loop.stats.decisionsPerSecond).toBe(1);
  });
  it("starts a new round when the game is won", async () => {
    const loop = new GameLoop(new StubAgent(), settings);
    loop.game.won = true;
    await loop.tick();
    expect(loop.stats.round).toBe(2);
    expect(loop.game.won).toBe(false);
  });
});
