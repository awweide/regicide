from __future__ import annotations

import argparse

from .engine import Game, Phase


def parse_slots(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a simple solo game of Regicide.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible games.")
    args = parser.parse_args()
    game = Game.new(seed=args.seed)
    print("Enter slot numbers separated by spaces. Type 'quit' to exit.")
    while game.phase not in {Phase.WON, Phase.LOST}:
        print()
        print(game.render())
        prompt = "Play slots> " if game.phase == Phase.PLAY else "Discard slots> "
        raw = input(prompt).strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            return
        try:
            slots = parse_slots(raw)
            if game.phase == Phase.PLAY:
                game.play_slots(slots)
            else:
                game.discard_slots(slots)
        except ValueError as exc:
            print(f"Invalid input: {exc}")
    print()
    print(game.render())


if __name__ == "__main__":
    main()
