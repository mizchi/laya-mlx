/**
 * Panel rendering for the browser chess demo: candidate probabilities, the
 * blunder shield, timing stats and the move list. No game logic lives here —
 * only drawing, mirroring ../snake/view.ts's panel half.
 */
import { $ } from "../dom.ts";
import type { Decision } from "./policy.ts";

export interface PanelStats {
  interventions: number;
}

export interface PanelLabels {
  engine: string;
  state: string;
  status: string;
  elapsedSeconds: number;
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

export class PanelView {
  render(
    decision: Decision | null,
    stats: PanelStats,
    labels: PanelLabels,
    history: string[],
  ): void {
    const list = $("candidates");
    list.innerHTML = "";
    for (const candidate of decision?.candidates ?? []) {
      const li = document.createElement("li");
      li.className = "candidate";
      if (decision?.proposed === candidate.san) li.classList.add("selected");
      if (candidate.best) li.classList.add("best");
      li.title = candidate.description;

      const san = document.createElement("span");
      san.className = "san";
      san.textContent = candidate.san;

      const track = document.createElement("span");
      track.className = "track";
      const bar = document.createElement("span");
      bar.className = "bar";
      const p = decision?.probabilities[candidate.san] ?? 0;
      bar.style.width = `${100 * p}%`;
      track.appendChild(bar);

      const val = document.createElement("span");
      val.className = "val";
      val.textContent = p.toFixed(2);

      li.append(san, track, val);
      list.appendChild(li);
    }

    $("executing").textContent = decision?.executed ?? "—";
    $("shield").hidden = !decision?.intervened;

    const danger = decision?.kingInDanger ?? 0;
    $("danger-bar").style.width = `${100 * danger}%`;
    $("danger-val").textContent = danger.toFixed(2);

    $("inference").textContent = `${(decision?.inferenceMs ?? 0).toFixed(1)} ms`;
    const plannerMs = decision ? decision.decisionMs - decision.inferenceMs : 0;
    $("planner").textContent = `${plannerMs.toFixed(1)} ms`;
    $("output-tokens").textContent = String(decision?.outputTokens ?? 0);

    $("engine").textContent = labels.engine;
    $("engine-title").textContent = labels.engine.split(" · ")[0] ?? labels.engine;

    $("interventions").textContent = pad4(stats.interventions);

    const state = $("state");
    state.textContent = labels.state;
    state.classList.remove("green", "red");
    state.classList.add(labels.state === "GAME OVER" || labels.state === "ERROR" ? "red" : "green");

    $("status").textContent = labels.status;
    $("clock").textContent = clock(labels.elapsedSeconds);

    const moves = $("moves");
    moves.innerHTML = "";
    for (let i = 0; i < history.length; i += 2) {
      const li = document.createElement("li");
      const moveNumber = i / 2 + 1;
      const white = history[i];
      const black = history[i + 1];
      li.textContent = black ? `${moveNumber}. ${white} ${black}` : `${moveNumber}. ${white}`;
      moves.appendChild(li);
    }
  }
}
