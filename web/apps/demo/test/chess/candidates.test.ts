import { Chess } from "chess.js";
import { describe, expect, it } from "vitest";

import { MATE, describeMove, evaluateMaterial, shortlist } from "../../src/chess/candidates.ts";

describe("evaluateMaterial", () => {
  it("counts pawns 1, minors 3, rooks 5, queens 9 from the side to move", () => {
    expect(evaluateMaterial(new Chess())).toBe(0);
    expect(evaluateMaterial(new Chess("k7/8/8/8/8/8/8/K6Q w - - 0 1"))).toBe(9);
    expect(evaluateMaterial(new Chess("k7/8/8/8/8/8/8/K6Q b - - 0 1"))).toBe(-9);
  });
});

describe("shortlist", () => {
  it("finds mate in one and marks it best", () => {
    const c = shortlist(new Chess("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"));
    expect(c[0]!.san).toBe("Ra8#");
    expect(c[0]!.score).toBe(MATE);
    expect(c[0]!.best).toBe(true);
    expect(c[0]!.description).toBe("Checkmate. Best.");
  });
  it("prefers winning a queen and describes the capture", () => {
    const c = shortlist(new Chess("q6k/8/8/8/8/8/8/R6K w - - 0 1"));
    expect(c[0]!.san).toBe("Rxa8+");
    expect(c[0]!.description).toMatch(/^Captures the queen\. Gives check\. Safe\. Best\.$/);
  });
  it("flags a move that hangs a piece and does not mark it best", () => {
    const c = shortlist(new Chess("4k3/8/4p3/8/8/2N5/8/4K3 w - - 0 1"), 20);
    const nd5 = c.find((x) => x.san === "Nd5");
    expect(nd5).toBeDefined();
    expect(nd5!.description).toBe("Hangs the knight to a pawn. Loses material.");
    expect(nd5!.best).toBe(false);
  });
  it("marks moves that allow mate in one", () => {
    const c = shortlist(new Chess("6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1"), 20);
    const losing = c.filter((x) => x.allowsMate);
    expect(losing.length).toBeGreaterThan(0);
    expect(losing.every((x) => x.score <= -MATE + 1)).toBe(true);
    expect(losing[0]!.description).toBe("Allows checkmate. Unsafe.");
  });
  it("returns at most six candidates ordered by score with exactly one best", () => {
    const c = shortlist(new Chess());
    expect(c.length).toBe(6);
    expect(c.map((x) => x.score)).toEqual([...c.map((x) => x.score)].sort((a, b) => b - a));
    expect(c.filter((x) => x.best)).toHaveLength(1);
    expect(c[0]!.description.endsWith(" Best.")).toBe(true);
    expect(c.slice(1).every((x) => x.description.endsWith("Safe."))).toBe(true);
  });
  it("describes castling, checks and development", () => {
    const castle = shortlist(new Chess("4k2r/8/8/8/8/8/8/4K2R w K - 0 1"), 30);
    expect(castle.some((x) => x.description.startsWith("Castles kingside."))).toBe(true);
    const c = new Chess();
    const nf3 = c.moves({ verbose: true }).find((m) => m.san === "Nf3")!;
    expect(describeMove(c, nf3, 0, false, null)).toBe("Develops the knight. Safe.");
    expect(c.fen()).toBe(new Chess().fen()); // board restored
  });
  it("leaves the board untouched after scoring", () => {
    const c = new Chess("q6k/8/8/8/8/8/8/R6K w - - 0 1");
    const fen = c.fen();
    shortlist(c);
    expect(c.fen()).toBe(fen);
  });
});
