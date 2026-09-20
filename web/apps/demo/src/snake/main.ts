/**
 * Wires the Snake page together: reads URL params, loads the model (or the
 * stub agent), drives `GameLoop`, and repaints through `SnakeView`. Ported
 * pacing behaviour from ../../../laya_mlx/snake/cli.py (a 1s pause at the end
 * of a round so the final board stays visible before the next one starts).
 */
import { loadDemoAgent, type LoaderUi } from "../loader.ts";
import { GameLoop, type LoopSettings } from "./loop.ts";
import type { PromptKind, SnakeAgent } from "./policy.ts";
import { StubAgent } from "./stub-agent.ts";
import { SnakeView, type GameState } from "./view.ts";

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function start(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const width = Number(params.get("width") ?? 24);
  const height = Number(params.get("height") ?? 16);
  const seed = Number(params.get("seed") ?? 7);
  const initialLength = Number(params.get("length") ?? 6);
  const fps = Number(params.get("fps") ?? 12);
  const guarded = !params.has("unassisted");
  const prompt: PromptKind = params.get("prompt") === "detailed" ? "detailed" : "compact";
  const startMaxSpeed = params.has("max-speed");
  const modelParam = params.get("model");

  const loadingEl = $("loading");
  const ui: LoaderUi = {
    status(text: string) {
      loadingEl.textContent = text;
    },
  };

  let agent: SnakeAgent;
  let engine: string;
  if (modelParam === "stub") {
    ui.status("stub agent (no model)");
    agent = new StubAgent();
    engine = "STUB · planner";
  } else {
    const loaded = await loadDemoAgent(ui);
    agent = loaded.agent;
    const providerLabel = loaded.provider === "webgpu" ? "WebGPU" : "wasm";
    const dtypeLabel = loaded.bundle.onnxConfig.dtype === "float16" ? "FP16" : "FP32";
    engine = `${providerLabel} · ${dtypeLabel}`;
  }

  const canvas = document.getElementById("board");
  if (!(canvas instanceof HTMLCanvasElement)) throw new Error("Missing canvas#board");
  const view = new SnakeView(canvas, width, height);

  const settings: LoopSettings = { width, height, seed, initialLength, fps, guarded, prompt };
  const loop = new GameLoop(agent, settings);
  loop.maxSpeed = startMaxSpeed;

  const pauseBtn = $("pause");
  const maxSpeedBtn = $("max-speed");
  maxSpeedBtn.setAttribute("aria-pressed", String(loop.maxSpeed));

  pauseBtn.addEventListener("click", () => loop.togglePause());
  $("slower").addEventListener("click", () => loop.changeSpeed(-2));
  $("faster").addEventListener("click", () => loop.changeSpeed(2));
  maxSpeedBtn.addEventListener("click", () => {
    loop.maxSpeed = !loop.maxSpeed;
    maxSpeedBtn.setAttribute("aria-pressed", String(loop.maxSpeed));
  });
  $("reset").addEventListener("click", () => loop.reset());

  window.addEventListener("keydown", (event) => {
    if (event.code === "Space") {
      event.preventDefault();
      loop.togglePause();
    } else if (event.key === "ArrowUp" || event.key === "+") {
      event.preventDefault();
      loop.changeSpeed(2);
    } else if (event.key === "ArrowDown" || event.key === "-") {
      event.preventDefault();
      loop.changeSpeed(-2);
    } else if (event.key === "r" || event.key === "R") {
      event.preventDefault();
      loop.reset();
    }
  });

  function draw(): void {
    const state: GameState = loop.paused
      ? "PAUSED"
      : !loop.game.alive
        ? "GAME OVER"
        : loop.game.won
          ? "BOARD CLEAR"
          : "LIVE";
    view.render(loop.game.snapshot(), loop.lastDecision, loop.stats, {
      engine,
      state,
      elapsedSeconds: loop.elapsedSeconds,
    });
    document.body.dataset.ready = "1";
  }

  draw();
  for (;;) {
    const startedAt = performance.now();
    const finished = !loop.game.alive || loop.game.won;
    await loop.tick();
    draw();
    await sleep(finished ? 1000 : loop.paused ? 100 : loop.delayAfter(startedAt));
  }
}

start().catch((error: unknown) => {
  const message = `ERROR ${String(error)}`;
  $("loading").textContent = message;
  const state = $("state");
  state.textContent = "ERROR";
  state.classList.remove("green");
  state.classList.add("red");
  console.error(error);
});
