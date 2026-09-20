/**
 * Candidate move generation for the chess demo's stub planner: 2-ply material
 * scoring plus Snake-style plain-English descriptions. No DOM, no Laya —
 * pure functions over a chess.js `Chess` instance.
 */
import { Chess, type Color, type Move } from "chess.js";

export const MATE = 1000;
export const SHORTLIST = 6;

const PIECE_VALUES: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };

const PIECE_NAMES: Record<string, string> = {
  p: "pawn",
  n: "knight",
  b: "bishop",
  r: "rook",
  q: "queen",
  k: "king",
};

function pieceName(symbol: string): string {
  return PIECE_NAMES[symbol] ?? symbol;
}

function isBackRank(square: string, color: Color): boolean {
  const rank = square[1];
  return color === "w" ? rank === "1" : rank === "8";
}

function clamp(value: number, limit: number): number {
  return Math.max(-limit, Math.min(limit, value));
}

/** Material balance from the side to move's perspective: pawns 1, minors 3, rooks 5, queens 9. */
export function evaluateMaterial(chess: Chess): number {
  const turn = chess.turn();
  let total = 0;
  for (const row of chess.board()) {
    for (const cell of row) {
      if (!cell) continue;
      const value = PIECE_VALUES[cell.type] ?? 0;
      total += cell.color === turn ? value : -value;
    }
  }
  return total;
}

export interface ScoredMove {
  score: number;
  reply: Move | null;
  allowsMate: boolean;
}

// Matches the destination square right after the "x" in a capturing SAN
// string (e.g. "Nxe4", "exd5", "Qbxd7+", "exd8=Q#"); the piece letters and
// disambiguation before "x" don't matter for this.
const CAPTURE_DEST_RE = /x([a-h][1-8])/;

/**
 * Value of the position for US after the opponent's reply captured on
 * `square`, given that we may still have one immediate recapture there.
 * Only scans replies to the *specific* square the opponent just captured on
 * (not every capture we have available) — `chess.moves({ verbose: true })`
 * is ~15x more expensive than the plain SAN list per call, and building
 * `Move` objects for every candidate's every reply's every recapture check
 * blew the perf budget (>1s on a busy middlegame); grepping SAN strings for
 * "x<square>" keeps this to one cheap non-verbose `moves()` call per capturing
 * reply. Caller undoes the outer moves; this always restores its own.
 */
function bestRecapture(chess: Chess, standPat: number, square: string): number {
  let best = standPat;
  for (const san of chess.moves()) {
    const captureSquare = CAPTURE_DEST_RE.exec(san)?.[1];
    if (captureSquare !== square) continue;
    chess.move(san);
    try {
      let v: number;
      if (chess.isCheckmate()) {
        v = MATE;
      } else if (chess.isDraw()) {
        v = 0;
      } else {
        // It is the opponent's turn again after our recapture: negate their
        // perspective to get ours.
        v = -evaluateMaterial(chess);
      }
      if (v > best) best = v;
    } finally {
      chess.undo();
    }
  }
  return best;
}

/** Score a legal move by playing it, then the opponent's best (worst-for-us) reply. */
export function scoreMove(chess: Chess, move: Move): ScoredMove {
  // Play the candidate move; the try/finally below always undoes it before
  // we return, however this function exits.
  if (move.promotion) {
    chess.move({ from: move.from, to: move.to, promotion: move.promotion });
  } else {
    chess.move({ from: move.from, to: move.to });
  }
  try {
    if (chess.isCheckmate()) {
      return { score: MATE, reply: null, allowsMate: false };
    }
    if (chess.isDraw()) {
      return { score: 0, reply: null, allowsMate: false };
    }

    let worst = Infinity;
    let worstReply: Move | null = null;
    let allowsMate = false;
    for (const replyMove of chess.moves({ verbose: true })) {
      chess.move(replyMove.san);
      try {
        let value: number;
        let mates = false;
        if (chess.isCheckmate()) {
          value = -MATE;
          mates = true;
        } else if (chess.isDraw()) {
          value = 0;
        } else {
          // One-step capture quiescence: a piece that lands on a defended
          // square isn't actually lost, so let us recapture on it before
          // scoring the reply as a material loss (fixes the horizon effect
          // where a defended piece reads as "hung"). Only worth scanning for
          // a recapture when the reply itself captured something.
          const standPat = evaluateMaterial(chess);
          value = replyMove.isCapture() ? bestRecapture(chess, standPat, replyMove.to) : standPat;
        }
        if (value < worst) {
          worst = value;
          worstReply = replyMove;
          allowsMate = mates;
        }
      } finally {
        chess.undo();
      }
    }

    let bonus = 0;
    if (chess.inCheck()) bonus += 0.1;
    // Hand-tuned nudges to break material ties; smaller than a pawn on purpose.
    if (move.isKingsideCastle() || move.isQueensideCastle()) bonus += 0.3;
    if ((move.piece === "n" || move.piece === "b") && isBackRank(move.from, move.color))
      bonus += 0.2;
    if (
      move.piece === "p" &&
      (move.to === "d4" || move.to === "e4" || move.to === "d5" || move.to === "e5")
    ) {
      bonus += 0.2;
    }

    const worstValue = worst === Infinity ? 0 : worst;
    return { score: worstValue + bonus, reply: worstReply, allowsMate };
  } finally {
    chess.undo();
  }
}

/** Describe a move (board must be BEFORE the move) in the Snake-style "sentence + Safe/Loses material" form. */
export function describeMove(
  chess: Chess,
  move: Move,
  materialDelta: number,
  allowsMate: boolean,
  reply: Move | null,
): string {
  if (move.san.endsWith("#")) return "Checkmate. Best.";
  if (allowsMate) return "Allows checkmate. Unsafe.";

  const parts: string[] = [];
  const isCapture = move.isCapture();
  let hangs = false;

  if (isCapture) {
    parts.push(`Captures the ${pieceName(move.captured ?? "")}.`);
  } else if (Math.round(materialDelta) <= -1 && reply && reply.isCapture()) {
    hangs = true;
    const attacker = pieceName(reply.piece);
    if (reply.to === move.to && move.piece !== "k") {
      // The reply recaptures on the square we just moved to: our own piece hangs.
      // (The king itself can never legally be the piece captured, since a reply
      // that "recaptures" the king would mean the position was already illegal.)
      parts.push(`Hangs the ${pieceName(move.piece)} to a ${attacker}.`);
    } else {
      // The reply captures elsewhere: an unrelated piece was left en prise.
      parts.push(`Drops the ${pieceName(reply.captured ?? "")} to a ${attacker}.`);
    }
  }

  if (move.isKingsideCastle()) parts.push("Castles kingside.");
  else if (move.isQueensideCastle()) parts.push("Castles queenside.");

  if (move.san.includes("+")) parts.push("Gives check.");

  if (
    !isCapture &&
    !hangs &&
    (move.piece === "n" || move.piece === "b") &&
    isBackRank(move.from, move.color)
  ) {
    parts.push(`Develops the ${pieceName(move.piece)}.`);
  }

  if (parts.length === 0) parts.push("Quiet move.");
  parts.push(materialDelta < 0 ? "Loses material." : "Safe.");
  return parts.join(" ");
}

export interface Candidate {
  san: string;
  from: string;
  to: string;
  score: number;
  description: string;
  best: boolean;
  allowsMate: boolean;
}

/** Score every legal move, keep the top `limit` by score, and describe each. */
export function shortlist(chess: Chess, limit: number = SHORTLIST): Candidate[] {
  const before = evaluateMaterial(chess);
  const moves = chess.moves({ verbose: true });

  const scored = moves.map((move) => {
    const { score, reply, allowsMate } = scoreMove(chess, move);
    const materialDelta = clamp(score, MATE) - before;
    return { move, score, reply, allowsMate, materialDelta };
  });

  scored.sort((a, b) => b.score - a.score);

  return scored.slice(0, limit).map((item, index) => {
    let description = describeMove(
      chess,
      item.move,
      item.materialDelta,
      item.allowsMate,
      item.reply,
    );
    const best = index === 0;
    if (best && !description.endsWith("Best.")) {
      description = `${description} Best.`;
    }
    return {
      san: item.move.san,
      from: item.move.from,
      to: item.move.to,
      score: item.score,
      description,
      best,
      allowsMate: item.allowsMate,
    };
  });
}
