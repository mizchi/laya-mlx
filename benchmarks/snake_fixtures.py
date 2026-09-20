"""Dump Snake rule outcomes and prompts for the browser port (web/apps/demo/src/snake).

A seeded game is played with the planner's best safe direction, so no model is
needed. Each step records the board before the move, every MoveInfo, food
reachability, the compact and detailed prompts, and the board after the move.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from laya_mlx.snake.game import SnakeGame, hamiltonian_cycle
from laya_mlx.snake.policy import build_prompt

ROOT = Path(__file__).resolve().parents[1]


def play(width, height, seed, initial_length, steps):
    game = SnakeGame(width, height, seed, initial_length)
    records = []
    for _ in range(steps):
        if not game.alive or game.won:
            break
        moves = game.moves()
        reachable, space = game.food_reachability()
        prompts = {
            kind: dict(
                zip(
                    ("state", "questions", "preferred"),
                    build_prompt(game, moves, reachable, space, kind),
                )
            )
            for kind in ("compact", "detailed")
        }
        before = game.snapshot()
        safe = [m for m in moves if m.safe]
        direction = max(safe, key=lambda m: m.advance).direction
        ate = game.step(direction)
        records.append(
            {
                "before": before,
                "moves": [asdict(m) for m in moves],
                "food_reachable": reachable,
                "open_cells": space,
                "prompts": prompts,
                "executed": direction,
                "ate": ate,
                "after": game.snapshot(),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "web/fixtures/snake-rules.json")
    args = parser.parse_args()
    games = [
        {"width": 24, "height": 16, "seed": 7, "initial_length": 6, "steps": 120},
        {"width": 8, "height": 6, "seed": 3, "initial_length": 4, "steps": 60},
        {"width": 4, "height": 4, "seed": 1, "initial_length": 2, "steps": 40},
    ]
    fixture = {
        # 5x4 has no game; it covers the odd-height transpose branch of hamiltonian_cycle.
        "cycles": {
            f"{w}x{h}": hamiltonian_cycle(w, h) for w, h in ((24, 16), (8, 6), (4, 4), (5, 4))
        },
        "games": [{**g, "records": play(**g)} for g in games],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, ensure_ascii=False, allow_nan=False) + "\n")
    steps = sum(len(g["records"]) for g in fixture["games"])
    print(f"{len(games)} games, {steps} steps -> {args.output}")


if __name__ == "__main__":
    main()
