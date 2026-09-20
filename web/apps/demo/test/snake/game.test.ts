import { describe, expect, it } from "vitest";

import { SnakeGame, hamiltonianCycle } from "../../src/snake/game.ts";
import { mulberry32 } from "../../src/snake/rng.ts";
import { snakeFixture } from "./fixtures.ts";

describe("hamiltonianCycle", () => {
  it.each(Object.entries(snakeFixture.cycles))("%s matches Python", (name, cells) => {
    const [w, h] = name.split("x").map(Number) as [number, number];
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
  it("plays a tiny board to the win and refuses to step afterwards", () => {
    const tiny = new SnakeGame(4, 4, 1, 15);
    while (tiny.alive && !tiny.won) {
      const safe = tiny.moves().filter((m) => m.safe);
      expect(safe.length).toBeGreaterThan(0);
      tiny.step(safe.reduce((a, b) => (b.advance > a.advance ? b : a)).direction);
    }
    expect(tiny.won).toBe(true);
    expect(tiny.food).toBeNull();
    expect(() => tiny.step("UP")).toThrow(/finished/);
  });
  it("reports wall, reverse and body deaths", () => {
    const game = new SnakeGame(8, 6, 3, 4);
    const reasons = new Set(["legal", "wall", "reverse", "body"]);
    for (const d of ["UP", "DOWN", "LEFT", "RIGHT"] as const)
      expect(reasons.has(game.legalReason(d))).toBe(true);
    const unsafe = game.moves().find((m) => !m.legal);
    if (unsafe) {
      expect(game.step(unsafe.direction)).toBe(false);
      expect(game.alive).toBe(false);
      expect(game.deathReason).toBe(unsafe.reason);
    }
  });
});
