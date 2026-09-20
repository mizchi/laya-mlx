/**
 * Canvas + panel rendering for the browser Snake demo. Mirrors the layout of
 * ../../../laya_mlx/snake/ui.py's `compose`: a board on the left and a stats
 * panel on the right. No game logic lives here — only drawing.
 */
import { DIRECTIONS, type Snapshot } from "./game.ts";
import type { LoopStats } from "./loop.ts";
import type { Decision } from "./policy.ts";

export type GameState = "LOADING" | "LIVE" | "PAUSED" | "GAME OVER" | "BOARD CLEAR";

export interface ViewLabels {
  engine: string;
  state: GameState;
  elapsedSeconds: number;
}

const BOARD_BG = "#0b1216";
const DOT_COLOR = "#13272e";
const HEAD_COLOR = "#dcfff0";
const FOOD_COLOR = "#f5c26b";

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el;
}

function pad2(n: number): string {
  return n.toString().padStart(2, "0");
}

function pad4(n: number): string {
  return n.toString().padStart(4, "0");
}

function clock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  return `${pad2(Math.floor(total / 60))}:${pad2(total % 60)}`;
}

function bodyColor(index: number, length: number): string {
  if (index === 0) return HEAD_COLOR;
  const fraction = 1 - index / Math.max(1, length);
  const r = Math.round(18 + 64 * fraction);
  const g = Math.round(73 + 150 * fraction);
  const b = Math.round(57 + 102 * fraction);
  return `rgb(${r}, ${g}, ${b})`;
}

export class SnakeView {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly cell: number;
  private readonly cols: number;
  private readonly rows: number;

  constructor(canvas: HTMLCanvasElement, width: number, height: number) {
    this.cols = width;
    this.rows = height;
    this.cell = Math.max(1, Math.floor(Math.min(canvas.width / width, canvas.height / height)));
    canvas.width = this.cell * width;
    canvas.height = this.cell * height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    this.ctx = ctx;
  }

  private drawBoard(board: Snapshot): void {
    const { ctx, cell } = this;
    const w = cell * this.cols;
    const h = cell * this.rows;
    ctx.fillStyle = BOARD_BG;
    ctx.fillRect(0, 0, w, h);

    ctx.fillStyle = DOT_COLOR;
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        const cx = x * cell + cell / 2;
        const cy = y * cell + cell / 2;
        ctx.fillRect(cx - 1, cy - 1, 2, 2);
      }
    }

    const length = board.body.length;
    for (let index = length - 1; index >= 0; index--) {
      const [x, y] = board.body[index]!;
      ctx.fillStyle = bodyColor(index, length);
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }

    if (board.food) {
      const [fx, fy] = board.food;
      ctx.fillStyle = FOOD_COLOR;
      ctx.beginPath();
      ctx.arc(fx * cell + cell / 2, fy * cell + cell / 2, cell * 0.3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  render(board: Snapshot, decision: Decision | null, stats: LoopStats, labels: ViewLabels): void {
    this.drawBoard(board);

    const state = $("state");
    state.textContent = labels.state;
    state.classList.remove("green", "red");
    state.classList.add(board.alive ? "green" : "red");

    $("round").textContent = `ROUND ${pad2(stats.round)}`;
    $("score").textContent = String(board.score);
    $("length").textContent = String(board.length);
    $("best").textContent = String(stats.best);

    const capacity = board.width * board.height;
    const pct = capacity > 0 ? (100 * board.length) / capacity : 0;
    $("fill-bar").style.width = `${pct}%`;
    $("fill-pct").textContent = `${pct.toFixed(1)}%`;

    const probabilities = decision?.probabilities;
    for (const direction of DIRECTIONS) {
      const li = document.querySelector<HTMLElement>(`#probs li[data-dir="${direction}"]`);
      if (!li) throw new Error(`Missing element #probs li[data-dir="${direction}"]`);
      const p = probabilities?.[direction] ?? 0;
      const bar = li.querySelector<HTMLElement>(".bar");
      const val = li.querySelector<HTMLElement>(".val");
      if (!bar || !val) throw new Error(`Malformed #probs row for ${direction}`);
      bar.style.width = `${100 * p}%`;
      val.textContent = p.toFixed(2);
      li.classList.toggle("selected", decision?.proposed === direction);
    }

    $("executing").textContent = decision?.executed ?? "—";
    const shield = $("shield");
    shield.hidden = !decision?.intervened;

    const risk = decision?.deadEndRisk ?? 0;
    const riskBar = $("risk-bar");
    riskBar.style.width = `${100 * risk}%`;
    riskBar.classList.remove("amber", "red");
    riskBar.classList.add(risk < 0.5 ? "amber" : "red");
    $("risk-val").textContent = risk.toFixed(2);

    const food = decision?.foodReachable ?? 0;
    $("food-bar").style.width = `${100 * food}%`;
    $("food-val").textContent = food.toFixed(2);

    $("inference").textContent = `${(decision?.inferenceMs ?? 0).toFixed(1)} ms`;
    $("rate").textContent = `${stats.decisionsPerSecond.toFixed(1)} /s`;
    $("output-tokens").textContent = String(decision?.outputTokens ?? 0);

    $("engine").textContent = labels.engine;
    $("engine-title").textContent = labels.engine.split(" · ")[0] ?? labels.engine;

    $("interventions").textContent = pad4(stats.interventions);
    $("clock").textContent = clock(labels.elapsedSeconds);
  }
}
