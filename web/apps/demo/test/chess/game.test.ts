import { describe, expect, it } from "vitest";

import { ChessGame } from "../../src/chess/game.ts";

describe("ChessGame", () => {
  it("starts with white to move and the user as white by default", () => {
    const g = new ChessGame();
    expect(g.userColor).toBe("w");
    expect(g.turn).toBe("w");
    expect(g.isUserTurn).toBe(true);
    expect(g.status).toEqual({ over: false, text: "White to move" });
  });
  it("applies SAN and from/to moves, records history, flips the turn", () => {
    const g = new ChessGame();
    expect(g.applyMove({ from: "e2", to: "e4" })).toBe("e4");
    expect(g.applyMove("e5")).toBe("e5");
    expect(g.history).toEqual(["e4", "e5"]);
    expect(g.turn).toBe("w");
    expect(g.applyMove({ from: "e2", to: "e4" })).toBeNull();
  });
  it("lists legal targets for a square and promotes to a queen by default", () => {
    const g = new ChessGame("8/P7/8/8/7k/8/8/4K3 w - - 0 1");
    expect(g.legalTargets("a7")).toEqual(["a8"]);
    expect(g.applyMove({ from: "a7", to: "a8" })).toBe("a8=Q");
  });
  it("reports checkmate, stalemate and draws", () => {
    const mate = new ChessGame("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1");
    mate.applyMove("Ra8#");
    expect(mate.status).toEqual({ over: true, text: "Checkmate — White wins" });
    const stalemate = new ChessGame("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    expect(stalemate.status).toEqual({ over: true, text: "Stalemate — draw" });
    const insufficient = new ChessGame("8/8/8/8/8/8/8/k6K w - - 0 1");
    expect(insufficient.status.over).toBe(true);
  });
  it("switches sides so the user plays black", () => {
    const g = new ChessGame(undefined, "b");
    expect(g.isUserTurn).toBe(false);
    g.applyMove("e4");
    expect(g.isUserTurn).toBe(true);
  });
  it("exposes the board for rendering", () => {
    const g = new ChessGame();
    const board = g.board();
    expect(board).toHaveLength(8);
    expect(board[7]![4]).toEqual({ square: "e1", type: "k", color: "w" });
    expect(board[3]![3]).toBeNull();
  });
  it("reports the last move's from/to squares, or null before any move", () => {
    const g = new ChessGame();
    expect(g.lastMove).toBeNull();
    g.applyMove({ from: "e2", to: "e4" });
    expect(g.lastMove).toEqual({ from: "e2", to: "e4" });
    g.applyMove("e5");
    expect(g.lastMove).toEqual({ from: "e7", to: "e5" });
  });
});
