/**
 * Shared HUD formatting helpers for the Snake and chess panels: zero-padded
 * numbers, a mm:ss clock, the engine label before " · ", and the green/red
 * state-class toggle. No game logic lives here — only formatting.
 */

export function pad2(n: number): string {
  return n.toString().padStart(2, "0");
}

export function pad4(n: number): string {
  return n.toString().padStart(4, "0");
}

export function formatClock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  return `${pad2(Math.floor(total / 60))}:${pad2(total % 60)}`;
}

/** The engine label's provider/model segment, before the first " · ". */
export function engineTitle(engine: string): string {
  return engine.split(" · ")[0] ?? engine;
}

/** Toggles the shared green/red state classes on an element. */
export function setStateClass(el: HTMLElement, ok: boolean): void {
  el.classList.remove("green", "red");
  el.classList.add(ok ? "green" : "red");
}
