/**
 * DOM board for the browser chess demo: a fixed grid of 64 `<button>` squares,
 * built once and re-labelled/re-ordered on every render. No chess logic lives
 * here beyond `game.legalTargets` — selection state, piece glyphs and square
 * highlighting only.
 */
import type { ChessGame } from "./game.ts";

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"] as const;
const RANKS = [8, 7, 6, 5, 4, 3, 2, 1] as const;

const WHITE_GLYPHS: Record<string, string> = {
  k: "♔",
  q: "♕",
  r: "♖",
  b: "♗",
  n: "♘",
  p: "♙",
};
const BLACK_GLYPHS: Record<string, string> = {
  k: "♚",
  q: "♛",
  r: "♜",
  b: "♝",
  n: "♞",
  p: "♟",
};
const PIECE_NAMES: Record<string, string> = {
  k: "king",
  q: "queen",
  r: "rook",
  b: "bishop",
  n: "knight",
  p: "pawn",
};

type Board = ReturnType<ChessGame["board"]>;

function pieceAt(board: Board, square: string): { type: string; color: "w" | "b" } | null {
  const file = FILES.indexOf(square[0] as (typeof FILES)[number]);
  const rank = Number(square[1]);
  const cell = board[8 - rank]?.[file];
  return cell ?? null;
}

export class BoardView {
  private readonly container: HTMLElement;
  private readonly onMove: (from: string, to: string) => void;
  private readonly squares = new Map<string, HTMLButtonElement>();
  private interactive = true;
  private selected: string | null = null;
  private targets = new Set<string>();
  private game: ChessGame | null = null;
  private flipped = false;

  constructor(container: HTMLElement, onMove: (from: string, to: string) => void) {
    this.container = container;
    this.onMove = onMove;
    for (const rank of RANKS) {
      for (const file of FILES) {
        const square = `${file}${rank}`;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "square";
        button.dataset.square = square;
        button.setAttribute("role", "gridcell");
        button.addEventListener("click", () => this.handleClick(square));
        this.squares.set(square, button);
        container.appendChild(button);
      }
    }
  }

  setInteractive(interactive: boolean): void {
    this.interactive = interactive;
    for (const button of this.squares.values()) {
      button.disabled = !interactive;
    }
  }

  private handleClick(square: string): void {
    if (!this.interactive || !this.game) return;
    const game = this.game;

    if (this.selected && this.targets.has(square)) {
      const from = this.selected;
      this.selected = null;
      this.targets = new Set();
      this.onMove(from, square);
      return;
    }

    this.selected = null;
    this.targets = new Set();
    const piece = pieceAt(game.board(), square);
    if (piece && piece.color === game.userColor) {
      this.selected = square;
      this.targets = new Set(game.legalTargets(square));
    }
    this.render(game, this.flipped);
  }

  /** Re-orders the 64 buttons so rank 8 is on top (or rank 1, when flipped) and re-labels each. */
  render(game: ChessGame, flipped: boolean): void {
    this.game = game;
    this.flipped = flipped;

    const files = flipped ? [...FILES].reverse() : FILES;
    const ranks = flipped ? [...RANKS].reverse() : RANKS;
    for (const rank of ranks) {
      for (const file of files) {
        this.container.appendChild(this.squares.get(`${file}${rank}`)!);
      }
    }

    const lastMove = game.lastMove;
    const board = game.board();

    for (const rank of RANKS) {
      for (const file of FILES) {
        const square = `${file}${rank}`;
        const button = this.squares.get(square)!;
        const fileIndex = FILES.indexOf(file);
        const light = (fileIndex + rank) % 2 === 0;
        button.classList.toggle("light", light);
        button.classList.toggle("dark", !light);

        button.classList.toggle("selected", this.selected === square);
        button.classList.toggle("target", this.targets.has(square));
        button.classList.toggle(
          "last",
          lastMove !== null && (lastMove.from === square || lastMove.to === square),
        );

        const cell = board[8 - rank]?.[fileIndex] ?? null;
        if (cell) {
          const colorName = cell.color === "w" ? "white" : "black";
          const glyphs = cell.color === "w" ? WHITE_GLYPHS : BLACK_GLYPHS;
          button.textContent = glyphs[cell.type] ?? "";
          button.classList.toggle("white", cell.color === "w");
          button.classList.toggle("black", cell.color === "b");
          button.setAttribute("aria-label", `${colorName} ${PIECE_NAMES[cell.type] ?? cell.type}`);
        } else {
          button.textContent = "";
          button.classList.remove("white", "black");
          button.setAttribute("aria-label", square);
        }
      }
    }
  }
}
