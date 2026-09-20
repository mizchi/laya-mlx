/**
 * Owns the game, the pacing and the running statistics for the browser Snake
 * demo. Rendering and keyboard/UI wiring are someone else's job. Ported from
 * the terminal loop in ../../../laya_mlx/snake/cli.py: pause, ±2 fps within
 * [1, 240], R resets with the next seed (seed + round - 1), and the best
 * score / decisions-per-second stats are tracked across rounds. `tick()`
 * deliberately does NOT auto-start the next round once a round ends
 * (`finished` becomes true) — the caller decides when to call `reset()`,
 * which lets the final board stay on screen for as long as it wants before
 * the next round begins.
 */
import { SnakeGame } from "./game.ts";
import { decide, type Decision, type PolicyOptions, type SnakeAgent } from "./policy.ts";

export interface LoopSettings extends PolicyOptions {
  width: number;
  height: number;
  seed: number;
  initialLength: number;
  /** Paced decisions per second; ignored in max-speed mode. */
  fps: number;
}

export interface LoopStats {
  best: number;
  round: number;
  decisions: number;
  interventions: number;
  startedAt: number;
  /** Decisions completed during the last second of wall clock. */
  decisionsPerSecond: number;
}

/**
 * `tick()` rejects when the policy or model fails and leaves the game unchanged; the caller
 * decides whether to stop or retry. It also discards (rather than applies) a decision that was
 * still in flight when `reset()` ran, so a slow inference call can never step a game that the
 * user has already moved on from.
 */
export class GameLoop {
  game: SnakeGame;
  lastDecision: Decision | null = null;
  paused = false;
  maxSpeed = false;
  fps: number;
  /** Bumped by every `reset()`; lets `advance()` detect a reset that raced an in-flight decision. */
  generation = 0;
  readonly stats: LoopStats;

  private readonly agent: SnakeAgent;
  private readonly settings: LoopSettings;
  private inFlight: Promise<void> | null = null;
  private recent: number[] = [];

  constructor(agent: SnakeAgent, settings: LoopSettings) {
    this.agent = agent;
    this.settings = settings;
    this.fps = Number.isFinite(settings.fps) ? Math.min(240, Math.max(1, settings.fps)) : 12;
    this.game = this.newGame(settings.seed);
    this.stats = {
      best: 0,
      round: 1,
      decisions: 0,
      interventions: 0,
      startedAt: performance.now(),
      decisionsPerSecond: 0,
    };
  }

  private newGame(seed: number): SnakeGame {
    return new SnakeGame(
      this.settings.width,
      this.settings.height,
      seed,
      this.settings.initialLength,
    );
  }

  get elapsedSeconds(): number {
    return (performance.now() - this.stats.startedAt) / 1000;
  }

  /** True once the current round has ended (death or a cleared board); `tick()` is then a no-op. */
  get finished(): boolean {
    return !this.game.alive || this.game.won;
  }

  togglePause(): void {
    this.paused = !this.paused;
  }

  changeSpeed(delta: number): void {
    this.fps = Math.min(240, Math.max(1, this.fps + delta));
  }

  /** Next round with the next seed, like the terminal demo's R key. */
  reset(): void {
    this.stats.round += 1;
    this.generation += 1;
    this.game = this.newGame(this.settings.seed + this.stats.round - 1);
    this.lastDecision = null;
  }

  /** One decision + move. Concurrent calls share the in-flight decision instead of stacking. */
  tick(): Promise<void> {
    if (this.inFlight) return this.inFlight;
    this.inFlight = this.advance().finally(() => {
      this.inFlight = null;
    });
    return this.inFlight;
  }

  private async advance(): Promise<void> {
    if (this.paused) return;
    if (this.finished) return;
    const game = this.game;
    const options: PolicyOptions = { guarded: this.settings.guarded, prompt: this.settings.prompt };
    const decision = await decide(this.agent, game, options);
    // A reset() (e.g. the user pressing R) may have replaced `this.game` while we were awaiting
    // the model; the decision was made against a game that no longer exists, so it must not be
    // applied to whatever game is current now.
    if (game !== this.game) return;
    this.lastDecision = decision;
    game.step(decision.executed);
    this.stats.decisions += 1;
    if (decision.intervened) this.stats.interventions += 1;
    this.stats.best = Math.max(this.stats.best, this.game.score);
    const now = performance.now();
    this.recent.push(now);
    this.recent = this.recent.filter((t) => now - t <= 1000);
    this.stats.decisionsPerSecond = this.recent.length;
  }

  /** Milliseconds to wait before the next tick in paced mode; 0 in max-speed mode. */
  delayAfter(tickStartedAt: number): number {
    if (this.maxSpeed) return 0;
    return Math.max(0, 1000 / this.fps - (performance.now() - tickStartedAt));
  }
}
