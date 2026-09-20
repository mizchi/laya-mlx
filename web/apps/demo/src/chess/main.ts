/**
 * Wires the chess page together: reads URL params, loads the model (or the
 * stub agent), drives `ChessGame` through `decide`, and repaints through
 * `BoardView`/`PanelView`. Sibling to ../snake/main.ts — same shape.
 */
import { $ } from "../dom.ts";
import { loadDemoAgent, type LoaderUi } from "../loader.ts";
import { StubAgent } from "../stub-agent.ts";
import { BoardView } from "./board.ts";
import { ChessGame, type Color } from "./game.ts";
import { PanelView } from "./panel.ts";
import { decide, type ChessAgent, type Decision } from "./policy.ts";

function reportError(error: unknown): void {
  const message = `ERROR ${String(error)}`;
  $("loading").textContent = message;
  const state = $("state");
  state.textContent = "ERROR";
  state.classList.remove("green");
  state.classList.add("red");
  console.error(error);
}

async function start(): Promise<void> {
  const params = new URLSearchParams(location.search);
  let side: Color = params.get("side") === "b" ? "b" : "w";
  const guarded = !params.has("unassisted");
  const fen = params.get("fen") ?? undefined;
  const modelParam = params.get("model");

  const loadingEl = $("loading");
  const ui: LoaderUi = {
    status(text: string) {
      loadingEl.textContent = text;
    },
  };

  let agent: ChessAgent;
  let engine: string;
  if (modelParam === "stub") {
    ui.status("stub agent (no model)");
    agent = new StubAgent();
    engine = "STUB · planner";
  } else {
    const loaded = await loadDemoAgent(ui, 2);
    agent = loaded.agent;
    const providerLabel = loaded.provider === "webgpu" ? "WebGPU" : "wasm";
    const dtypeLabel = loaded.bundle.onnxConfig.dtype === "float16" ? "FP16" : "FP32";
    engine = `${providerLabel} · ${dtypeLabel}`;
  }

  let game = new ChessGame(fen, side);
  let lastDecision: Decision | null = null;
  let interventions = 0;
  let thinking = false;
  const startedAt = performance.now();

  const boardEl = $("board");
  const board = new BoardView(boardEl, (from, to) => {
    void handleUserMove(from, to);
  });
  const panel = new PanelView();
  const switchSidesBtn = $("switch-sides");

  function stateLabel(): string {
    return game.status.over ? "GAME OVER" : "LIVE";
  }

  function draw(): void {
    board.render(game, side === "b");
    panel.render(
      lastDecision,
      { interventions },
      {
        engine,
        state: stateLabel(),
        status: game.status.text,
        elapsedSeconds: (performance.now() - startedAt) / 1000,
      },
      game.history,
    );
    document.body.dataset.ready = "1";
  }

  async function aiMove(): Promise<void> {
    if (thinking) return;
    thinking = true;
    board.setInteractive(false);
    try {
      const decision = await decide(agent, game, { guarded });
      const san = game.applyMove(decision.executed);
      if (!san) throw new Error(`Chess: engine proposed an illegal move: ${decision.executed}`);
      interventions += decision.intervened ? 1 : 0;
      lastDecision = decision;
    } finally {
      thinking = false;
    }
    draw();
    board.setInteractive(game.isUserTurn);
  }

  async function runAiMove(): Promise<void> {
    try {
      await aiMove();
    } catch (error) {
      reportError(error);
    }
  }

  async function handleUserMove(from: string, to: string): Promise<void> {
    const san = game.applyMove({ from, to });
    if (!san) return;
    draw();
    if (game.status.over) {
      board.setInteractive(false);
      return;
    }
    await runAiMove();
  }

  function newGame(): void {
    game = new ChessGame(fen, side);
    lastDecision = null;
    draw();
    board.setInteractive(game.isUserTurn);
    if (!game.isUserTurn && !game.status.over) void runAiMove();
  }

  $("new-game").addEventListener("click", newGame);
  switchSidesBtn.addEventListener("click", () => {
    side = side === "w" ? "b" : "w";
    switchSidesBtn.textContent = side === "w" ? "Play as black" : "Play as white";
    newGame();
  });

  setInterval(() => {
    draw();
  }, 1000);

  draw();
  board.setInteractive(game.isUserTurn);
  if (!game.isUserTurn && !game.status.over) await runAiMove();
}

start().catch(reportError);
