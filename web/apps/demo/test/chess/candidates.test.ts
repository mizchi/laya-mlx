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
  it("describes an unrelated piece left en prise as dropped, not the king", () => {
    // White king on e1, knight on c3, black pawn on d4 attacks the knight.
    // Every king move leaves the knight hanging to the pawn; none of them
    // should ever be described as the king hanging.
    const c = shortlist(new Chess("4k3/8/8/8/3p4/2N5/8/4K3 w - - 0 1"), 20);
    const ke2 = c.find((x) => x.san === "Ke2");
    expect(ke2).toBeDefined();
    expect(ke2!.description).toBe("Drops the knight to a pawn. Loses material.");
    for (const x of c) {
      expect(x.description).not.toContain("Hangs the king");
    }
  });
  it("does not flag a quiet move when the threatened piece is defended (recapture quiescence)", () => {
    // White pawn e4 is defended by Nc3; black knight f6 attacks it. An
    // unrelated quiet move like h3 must not read as dropping the pawn: if
    // black plays ...Nxe4, white recaptures Nxe4 and comes out even.
    const fen = "4k3/8/5n2/8/4P3/2N5/7P/4K3 w - - 0 1";
    const board = new Chess(fen);
    const c = shortlist(board, 20);
    const h3 = c.find((x) => x.san === "h3");
    expect(h3).toBeDefined();
    expect(h3!.description).toBe("Quiet move. Safe.");
    expect(board.fen()).toBe(fen); // board restored despite the extra recapture nesting
  });
  it("credits a recapture that wins back more than it loses, keeping Safe", () => {
    // White pawn e4 defended by Nc3; black queen g4 could grab the pawn but
    // Nxe4 would then win the queen. Without quiescence the old scoring
    // would have charged the candidate move for losing the pawn (it isn't:
    // that branch is never actually the opponent's best reply).
    const fen = "4k3/8/8/8/4P1q1/2N5/8/4K3 w - - 0 1";
    const before = new Chess(fen);
    const beforeFen = before.fen();
    const c = shortlist(before, 20);
    const kd2 = c.find((x) => x.san === "Kd2");
    expect(kd2).toBeDefined();
    expect(kd2!.description).toBe("Quiet move. Safe.");
    expect(before.fen()).toBe(beforeFen); // board restored despite the extra recapture nesting
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
