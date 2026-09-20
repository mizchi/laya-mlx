import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import type { Direction, MoveInfo, Snapshot } from "../../src/snake/game.ts";

export interface SnakeRecord {
  before: Snapshot;
  moves: MoveInfo[];
  food_reachable: boolean;
  open_cells: number;
  prompts: Record<
    "compact" | "detailed",
    { state: string; questions: Record<string, unknown>; preferred: string }
  >;
  executed: Direction;
  ate: boolean;
  after: Snapshot;
}
export interface SnakeFixture {
  cycles: Record<string, [number, number][]>;
  games: {
    width: number;
    height: number;
    seed: number;
    initial_length: number;
    steps: number;
    records: SnakeRecord[];
  }[];
}

const path = fileURLToPath(
  new URL("../../../../fixtures/snake-multilingual.json", import.meta.url),
);
export const snakeFixture: SnakeFixture = JSON.parse(readFileSync(path, "utf8"));
