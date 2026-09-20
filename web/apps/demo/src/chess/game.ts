import { Chess, type Color, type Square } from "chess.js";

export type { Color, Square };
export interface MoveInput {
  from: string;
  to: string;
  promotion?: "q" | "r" | "b" | "n";
}
export interface GameStatus {
  over: boolean;
  text: string;
}

/** Thin, rule-complete wrapper over chess.js with the demo's notion of a user side. */
export class ChessGame {
  readonly chess: Chess;
  readonly userColor: Color;
  constructor(fen?: string, userColor: Color = "w") {
    this.chess = fen ? new Chess(fen) : new Chess();
    this.userColor = userColor;
  }
  get turn(): Color {
    return this.chess.turn();
  }
  get isUserTurn(): boolean {
    return this.turn === this.userColor && !this.status.over;
  }
  get history(): string[] {
    return this.chess.history();
  }
  get fen(): string {
    return this.chess.fen();
  }
  /** chess.js board(): rank 8 first; each cell `{square, type, color}` or null. */
  board() {
    return this.chess.board();
  }
  legalTargets(from: string): string[] {
    const targets = this.chess.moves({ square: from as Square, verbose: true }).map((m) => m.to);
    return [...new Set(targets)];
  }
  /** The last played move's from/to squares, or null before any move has been made. */
  get lastMove(): { from: string; to: string } | null {
    const last = this.chess.history({ verbose: true }).at(-1);
    return last ? { from: last.from, to: last.to } : null;
  }
  /** Apply a SAN string or a from/to pair (promotion defaults to queen). Returns the SAN or null if illegal. */
  applyMove(move: string | MoveInput): string | null {
    try {
      const played =
        typeof move === "string"
          ? this.chess.move(move)
          : this.chess.move({ from: move.from, to: move.to, promotion: move.promotion ?? "q" });
      return played.san;
    } catch {
      return null;
    }
  }
  get status(): GameStatus {
    const c = this.chess;
    const side = c.turn() === "w" ? "White" : "Black";
    const other = c.turn() === "w" ? "Black" : "White";
    if (c.isCheckmate()) return { over: true, text: `Checkmate — ${other} wins` };
    if (c.isStalemate()) return { over: true, text: "Stalemate — draw" };
    if (c.isThreefoldRepetition()) return { over: true, text: "Draw by repetition" };
    if (c.isInsufficientMaterial()) return { over: true, text: "Draw — insufficient material" };
    if (c.isDraw()) return { over: true, text: "Draw" };
    return { over: false, text: c.inCheck() ? `${side} to move — check` : `${side} to move` };
  }
}
