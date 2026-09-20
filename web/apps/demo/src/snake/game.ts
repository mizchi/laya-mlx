/**
 * Deterministic Snake rules and a separately identified cycle safety planner.
 *
 * Faithful TypeScript port of ../../../laya_mlx/snake/game.py. Food placement
 * uses `mulberry32` (see ./rng.ts), a different PRNG than Python's
 * `random.Random`, so the sequence of spawned food cells will not match the
 * Python reference. The rules themselves (legality, safety, reachability,
 * stepping) are ported exactly.
 */
import { mulberry32 } from "./rng.ts";

export type Direction = "UP" | "DOWN" | "LEFT" | "RIGHT";

export const DIRECTIONS: readonly Direction[] = ["UP", "DOWN", "LEFT", "RIGHT"] as const;

export const VECTORS: Record<Direction, readonly [number, number]> = {
  UP: [0, -1],
  DOWN: [0, 1],
  LEFT: [-1, 0],
  RIGHT: [1, 0],
};

export type Cell = [number, number];

export interface MoveInfo {
  direction: Direction;
  legal: boolean;
  safe: boolean;
  advance: number;
  reason: string;
  eats: boolean;
}

export interface Snapshot {
  width: number;
  height: number;
  seed: number;
  body: Cell[];
  food: Cell | null;
  score: number;
  length: number;
  ticks: number;
  alive: boolean;
  won: boolean;
  death_reason: string | null;
}

function cellKey(cell: Cell): number {
  // Collision-free for any board narrower than 100000 cells; boards here are at most a few hundred.
  return cell[0] * 100000 + cell[1];
}

function mod(a: number, m: number): number {
  return ((a % m) + m) % m;
}

/** Visit each square once with adjacent steps, including the closing edge. */
export function hamiltonianCycle(width: number, height: number): Cell[] {
  if (Math.min(width, height) < 4 || (width % 2 !== 0 && height % 2 !== 0)) {
    throw new Error("Board dimensions must be >= 4, with at least one even dimension");
  }
  if (height % 2 !== 0) {
    return hamiltonianCycle(height, width).map(([a, b]) => [b, a] as Cell);
  }
  const path: Cell[] = [[0, 0]];
  for (let y = 0; y < height; y++) {
    if (y % 2 === 0) {
      for (let x = 1; x < width; x++) path.push([x, y]);
    } else {
      for (let x = width - 1; x >= 1; x--) path.push([x, y]);
    }
  }
  for (let y = height - 1; y >= 1; y--) path.push([0, y]);
  return path;
}

export class SnakeGame {
  readonly width: number;
  readonly height: number;
  readonly seed: number;
  readonly cycle: Cell[];
  readonly capacity: number;

  private readonly indices: Map<number, number>;
  private readonly rng: () => number;

  body: Cell[];
  food: Cell | null;
  score: number;
  ticks: number;
  alive: boolean;
  won: boolean;
  deathReason: string | null;

  constructor(width = 24, height = 16, seed = 7, initialLength = 6) {
    this.width = width;
    this.height = height;
    this.seed = seed;
    this.cycle = hamiltonianCycle(width, height);
    this.indices = new Map();
    this.cycle.forEach((cell, index) => this.indices.set(cellKey(cell), index));
    this.capacity = width * height;
    if (!(initialLength >= 2 && initialLength < this.capacity)) {
      throw new Error("Initial length must be >= 2 and smaller than the board");
    }
    this.rng = mulberry32(seed);
    const start = this.indices.get(cellKey([Math.floor(width / 2), Math.floor(height / 2)]))!;
    this.body = [];
    for (let i = 0; i < initialLength; i++) {
      const cell = this.cycle[mod(start - i, this.capacity)]!;
      this.body.push(cell);
    }
    this.score = 0;
    this.ticks = 0;
    this.alive = true;
    this.won = false;
    this.deathReason = null;
    this.food = this.spawnFood();
  }

  static fromSnapshot(snapshot: Snapshot): SnakeGame {
    const game = new SnakeGame(snapshot.width, snapshot.height, snapshot.seed, 2);
    game.body = snapshot.body.map((cell) => [cell[0], cell[1]] as Cell);
    game.food = snapshot.food ? ([snapshot.food[0], snapshot.food[1]] as Cell) : null;
    game.score = snapshot.score;
    game.ticks = snapshot.ticks;
    game.alive = snapshot.alive;
    game.won = snapshot.won;
    game.deathReason = snapshot.death_reason;
    return game;
  }

  get head(): Cell {
    return this.body[0]!;
  }

  private spawnFood(): Cell | null {
    const occupied = new Set(this.body.map(cellKey));
    const empty = this.cycle.filter((cell) => !occupied.has(cellKey(cell)));
    if (empty.length === 0) return null;
    return empty[Math.floor(this.rng() * empty.length)]!;
  }

  target(direction: Direction): Cell {
    const [dx, dy] = VECTORS[direction];
    const [hx, hy] = this.head;
    return [hx + dx, hy + dy];
  }

  legalReason(direction: Direction): string {
    const cell = this.target(direction);
    const [x, y] = cell;
    if (!(x >= 0 && x < this.width && y >= 0 && y < this.height)) return "wall";
    const second = this.body[1]!;
    if (cell[0] === second[0] && cell[1] === second[1]) return "reverse";
    const occupied = new Set(this.body.map(cellKey));
    if (!(this.food && cell[0] === this.food[0] && cell[1] === this.food[1])) {
      // The tail moves on a non-growing step.
      const tail = this.body[this.body.length - 1]!;
      occupied.delete(cellKey(tail));
    }
    return occupied.has(cellKey(cell)) ? "body" : "legal";
  }

  moves(): MoveInfo[] {
    if (!this.alive || this.won) return [];
    const headIndex = this.indices.get(cellKey(this.head))!;
    const tail = this.body[this.body.length - 1]!;
    const tailDistance = mod(this.indices.get(cellKey(tail))! - headIndex, this.capacity);
    const food = this.food!;
    const foodDistance = mod(this.indices.get(cellKey(food))! - headIndex, this.capacity);
    const moves: MoveInfo[] = [];
    for (const direction of DIRECTIONS) {
      let reason = this.legalReason(direction);
      const legal = reason === "legal";
      const target = this.target(direction);
      const targetIndex = this.indices.get(cellKey(target));
      const advance = mod((targetIndex ?? headIndex) - headIndex, this.capacity);
      const eats = target[0] === food[0] && target[1] === food[1];
      let safe = legal;
      if (safe && (advance > tailDistance || (advance === tailDistance && eats))) {
        safe = false;
        reason = "would cross the tail";
      }
      if (safe && (advance === 0 || advance > foodDistance)) {
        safe = false;
        reason = "would skip the food on the safe route";
      }
      moves.push({ direction, legal, safe, advance, reason, eats });
    }
    return moves;
  }

  /** Current empty-cell connectivity; the occupied tail is not treated as empty. */
  foodReachability(): [boolean, number] {
    const blocked = new Set(this.body.map(cellKey));
    blocked.delete(cellKey(this.head));
    const visited = new Set<number>([cellKey(this.head)]);
    const queue: Cell[] = [this.head];
    let cursor = 0;
    while (cursor < queue.length) {
      const [x, y] = queue[cursor]!;
      cursor++;
      for (const [dx, dy] of Object.values(VECTORS)) {
        const cell: Cell = [x + dx, y + dy];
        const key = cellKey(cell);
        if (
          cell[0] >= 0 &&
          cell[0] < this.width &&
          cell[1] >= 0 &&
          cell[1] < this.height &&
          !blocked.has(key) &&
          !visited.has(key)
        ) {
          visited.add(key);
          queue.push(cell);
        }
      }
    }
    const reachable = !!this.food && visited.has(cellKey(this.food));
    return [reachable, visited.size];
  }

  step(direction: Direction, placeFood: () => Cell | null = () => this.spawnFood()): boolean {
    if (!this.alive || this.won) throw new Error("Cannot step a finished game");
    this.ticks++;
    const reason = this.legalReason(direction);
    if (reason !== "legal") {
      this.alive = false;
      this.deathReason = reason;
      return false;
    }
    const target = this.target(direction);
    this.body.unshift(target);
    if (this.food && target[0] === this.food[0] && target[1] === this.food[1]) {
      this.score++;
      if (this.body.length === this.capacity) {
        this.won = true;
        this.food = null;
      } else {
        this.food = placeFood();
      }
      return true;
    }
    this.body.pop();
    return false;
  }

  snapshot(): Snapshot {
    return {
      width: this.width,
      height: this.height,
      seed: this.seed,
      body: this.body.map((cell) => [cell[0], cell[1]] as Cell),
      food: this.food ? ([this.food[0], this.food[1]] as Cell) : null,
      score: this.score,
      length: this.body.length,
      ticks: this.ticks,
      alive: this.alive,
      won: this.won,
      death_reason: this.deathReason,
    };
  }
}
