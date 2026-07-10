from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .cli import parse_slots
from .engine import CARD_VALUES, Game, Phase


@dataclass
class Ollama:
    model: str
    url: str = "http://localhost:11434"
    timeout: float = 20
    retries: int = 2

    def prompt(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self.url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        last = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode()).get("response", "")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"ollama gave up after {self.retries + 1} attempt(s): {last}")


def read_texts(folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    for path in sorted(folder.glob("*.txt")):
        parts.append(f"--- {path.name} ---\n{path.read_text(errors='replace')}")
    return "\n\n".join(parts) or "(no local context files yet)"


def score(game: Game, illegal_moves: int) -> int:
    enemies_defeated = 12 - len(game.enemy_pile) - (1 if game.active_enemy else 0)
    hand_value = sum(CARD_VALUES[card.rank] for card in game.hand if card is not None)
    return 10000 * enemies_defeated + 100 * hand_value + len(game.draw_pile) - 1000 * illegal_moves


def move_prompt(game: Game, context: str, illegal_moves: int, last_error: str = "") -> str:
    return f"""You are playing solo Regicide. Reply with only hand slot numbers, e.g. 1 3.
Use a legal move for the current phase.

Local context:
{context}

Illegal moves so far: {illegal_moves}
Last error: {last_error or 'none'}

Game state:
{game.render()}
"""


def revise_prompt(context: str, result: dict) -> str:
    return f"""Revise your Regicide strategy notes for the next game.
Return only the complete new contents of strategy.txt.

Current text files:
{context}

Last game result JSON:
{json.dumps(result, indent=2)}
"""


def run_one(args: argparse.Namespace, ollama: Ollama, game_no: int) -> dict:
    game = Game.new(seed=args.seed + game_no if args.seed is not None else None)
    illegal = 0
    last_error = ""
    log_path = args.log_dir / f"game-{int(time.time())}-{game_no}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        turn = 0
        while game.phase not in {Phase.WON, Phase.LOST} and illegal < args.max_illegal:
            context = read_texts(args.context_dir)
            before = game.render()
            try:
                response = ollama.prompt(move_prompt(game, context, illegal, last_error))
                slots = parse_slots(response)
                if game.phase == Phase.PLAY:
                    game.play_slots(slots)
                else:
                    game.discard_slots(slots)
                last_error = ""
            except Exception as exc:  # keep games moving; illegal engine moves and ollama failures both count
                response = locals().get("response", "")
                slots = []
                illegal += 1
                last_error = str(exc)
            turn += 1
            log.write(json.dumps({
                "turn": turn,
                "phase_before": before,
                "response": response,
                "slots": slots,
                "illegal_moves": illegal,
                "error": last_error,
                "phase_after": game.render(),
            }) + "\n")
    if illegal >= args.max_illegal and game.phase not in {Phase.WON, Phase.LOST}:
        game.phase = Phase.LOST
        game.message = f"Stopped after {illegal} illegal move(s)."
    result = {"game": game_no, "phase": game.phase.value, "score": score(game, illegal), "illegal_moves": illegal, "log": str(log_path)}
    print(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Regicide with a local Ollama LLM agent.")
    parser.add_argument("--model", default="llama3", help="Ollama model name.")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--context-dir", type=Path, default=Path("agent_context"))
    parser.add_argument("--log-dir", type=Path, default=Path("agent_logs"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-illegal", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--revise-between", action="store_true", help="After each game, ask Ollama to rewrite context-dir/strategy.txt.")
    args = parser.parse_args()
    ollama = Ollama(args.model, args.ollama_url, args.timeout, args.retries)
    for game_no in range(1, args.games + 1):
        result = run_one(args, ollama, game_no)
        if args.revise_between:
            text = ollama.prompt(revise_prompt(read_texts(args.context_dir), result))
            args.context_dir.mkdir(parents=True, exist_ok=True)
            (args.context_dir / "strategy.txt").write_text(text)


if __name__ == "__main__":
    main()
