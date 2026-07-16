from __future__ import annotations

import argparse
import inspect
import json
import random
import shutil
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .cli import parse_slots
from .engine import CARD_VALUES, Game, Phase


def limit_words(text: str, max_words: int = 200) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words])


def parse_agent_response(response: str) -> tuple[list[int], str, str]:
    sections: dict[str, list[str]] = {"move": [], "comment": [], "memory": []}
    current: str | None = None
    for raw_line in response.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith(("1:", "move:")):
            current = "move"
            sections[current].append(line.split(":", 1)[1].strip())
        elif lowered.startswith(("2:", "comment:")):
            current = "comment"
            sections[current].append(line.split(":", 1)[1].strip())
        elif lowered.startswith(("3:", "memory:")):
            current = "memory"
            sections[current].append(line.split(":", 1)[1].strip())
        elif current is not None:
            sections[current].append(line)

    move_text = "\n".join(part for part in sections["move"] if part).strip()
    comment = limit_words("\n".join(part for part in sections["comment"] if part))
    memory = limit_words("\n".join(part for part in sections["memory"] if part))
    return parse_slots(move_text), comment, memory


def _read_ollama_json(url: str, timeout: float) -> object:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


@dataclass
class Ollama:
    model: str
    url: str = "http://localhost:11434"
    timeout: float = 300
    retries: int = 0
    num_predict: int = 25000
    temperature: float = 0.2
    think: bool = True
    stream: bool = False

    def check_connection(self, progress=print) -> dict:
        """Verify basic Ollama HTTP and generation communication with progress output."""
        base_url = self.url.rstrip("/")
        result: dict[str, object] = {"url": base_url, "model": self.model}

        progress(f"[1/4] Checking Ollama server URL: {base_url}")
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError(f"invalid Ollama URL: {base_url!r}")

        progress(f"[2/4] Requesting model list from {base_url}/api/tags")
        tags = _read_ollama_json(f"{base_url}/api/tags", self.timeout)
        models = []
        if isinstance(tags, dict):
            for item in tags.get("models", []):
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    models.append(item["name"])
        result["available_models"] = models
        if models:
            progress(f"      Found {len(models)} model(s): {', '.join(models)}")
        else:
            progress("      Server responded, but no installed models were reported.")

        progress(f"[3/4] Sending a minimal non-streaming /api/generate prompt to model {self.model!r}")
        started = time.monotonic()
        response = self.prompt("Reply with exactly: pong", progress=progress)
        elapsed = time.monotonic() - started
        result["response"] = response
        result["elapsed_seconds"] = round(elapsed, 3)
        progress(f"      Received response in {elapsed:.2f}s: {response!r}")

        progress("[4/4] Ollama communication check completed successfully.")
        return result

    def prompt(self, prompt: str, progress=None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": self.stream,
            "think": self.think,
            "options": {
                "num_predict": self.num_predict,
                "temperature": self.temperature,
            },
        }
    
        body = json.dumps(payload).encode()
        last = None
    
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            req = urllib.request.Request(
                f"{self.url.rstrip('/')}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            try:
                if progress is not None:
                    progress(
                        f"      Ollama attempt {attempt + 1}/{self.retries + 1} "
                        f"(timeout={self.timeout:g}s, num_predict={self.num_predict}, "
                        f"stream={self.stream})"
                    )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if not self.stream:
                        data = json.loads(resp.read().decode())
                        result = data.get("response", "")
    
                        if progress is not None:
                            progress(
                                f"      Ollama attempt completed in "
                                f"{time.monotonic() - started:.2f}s"
                            )
                        return result
    
                    # Streaming mode
                    response_parts = []
                    thinking_parts = []
                    
                    response_buffer = []
                    thinking_buffer = []
                    
                    last_flush = time.monotonic()
                    
                    for raw in resp:
                        if not raw:
                            continue
                    
                        chunk = json.loads(raw.decode())
                    
                        thinking = chunk.get("thinking")
                        if thinking:
                            thinking_parts.append(thinking)
                            thinking_buffer.append(thinking)
                    
                        text = chunk.get("response")
                        if text:
                            response_parts.append(text)
                            response_buffer.append(text)
                    
                        now = time.monotonic()
                        if progress is not None and now - last_flush > 0.25:
                    
                            if thinking_buffer:
                                progress("[thinking] " + "".join(thinking_buffer))
                                thinking_buffer.clear()
                    
                            if response_buffer:
                                progress("[response] " + "".join(response_buffer))
                                response_buffer.clear()
                    
                            last_flush = now
                    
                        if chunk.get("done"):
                            break
                    
                    # Flush anything left over.
                    if progress is not None:
                        if thinking_buffer:
                            progress("[thinking] " + "".join(thinking_buffer))
                        if response_buffer:
                            progress("[response] " + "".join(response_buffer))
                    
                    return "".join(response_parts)

            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                json.JSONDecodeError,
            ) as exc:
                last = exc
                if progress is not None:
                    progress(
                        f"      Ollama attempt failed after "
                        f"{time.monotonic() - started:.2f}s: {exc}"
                    )
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(f"ollama gave up after {self.retries + 1} attempt(s): {last}")


def text_paths(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    return sorted(folder.glob("*.txt"))


def read_texts(folder: Path) -> str:
    parts = []
    for path in text_paths(folder):
        parts.append(f"--- {path.name} ---\n{path.read_text(errors='replace')}")
    return "\n\n".join(parts) or "(no local context files yet)"


def create_run_dir(log_root: Path) -> Path:
    log_root.mkdir(parents=True, exist_ok=True)
    for i in range(1,1000):
        d=log_root/f"run_{i:03d}"
        if not d.exists():
            d.mkdir()
            return d
    raise RuntimeError("No free run directory.")


def prompt_ollama(ollama, prompt: str, progress=print) -> str:
    """Call an Ollama-like object, passing progress when its prompt method supports it."""
    prompt_func = ollama.prompt
    if "progress" in inspect.signature(prompt_func).parameters:
        return prompt_func(prompt, progress=progress)
    return prompt_func(prompt)


def game_seed(args: argparse.Namespace, game_no: int) -> int | None:
    if args.seed_mode == "random":
        return random.SystemRandom().randrange(0, 2**32)
    if args.seed is None:
        return None
    if args.seed_mode == "fixed":
        return args.seed
    return args.seed + game_no


def score(game: Game, illegal_moves: int) -> int:
    enemies_defeated = 12 - len(game.enemy_pile) - (1 if game.active_enemy else 0)
    hand_value = sum(CARD_VALUES[card.rank] for card in game.hand if card is not None)
    return 10000 * enemies_defeated + 100 * hand_value + len(game.draw_pile) - 1000 * illegal_moves


def move_prompt(game: Game, context: str, memory: str, illegal_moves: int, last_error: str = "") -> str:
    return f"""You are playing solo Regicide. Reply only in this exact three-line format:
1: <hand slot numbers>
2: <optional brief comment, up to 200 words, explaining the choice in the game log>
3: <short-term memory, up to 200 words, carried over to the next turn of the game>

For example, this is a good attempt at a valid response:
1: 1 7
2: Play spades against undamaged enemy to maximize value
3: Nothing important to remember
The following are also correctly formatted:
1: 5 (single card)
1: (no card: allowed only with empty hand)

This is the current game state:
{game.render()}

Short-term memory from previous turn:
{memory or '(none)'}

Use the rules and strategies to determine valid and good plays.
The comment is logged for later strategy revision but is not shown to future move prompts.
The memory is passed to your next move prompt in this same game; use it for facts like known top draw-pile cards or remaining enemies.

Rules for the game and self-discovered advice for how to play:
{context}

Illegal moves so far: {illegal_moves}
Last move error feedback (use this to fix your next response): {last_error or 'none'}

Repeating the key task:
Reply only in this exact three-line format:
1: <hand slot numbers>
2: <optional brief comment, up to 200 words, explaining the choice in the game log>
3: <short-term memory, up to 200 words, carried over to the next turn of the game>

This is the current game state:
{game.render()}

Short-term memory from previous turn:
{memory or '(none)'}
"""


def revise_prompt(context: str, result: dict) -> str:
    return f"""Revise strategy.txt based on how the previous game played out. Return only the complete new contents of strategy.txt.
Make sure to retain the useful parts of the old strategy.txt. Focus on improving expected score, by avoiding illegal moves and progressing by defeating more enemies.
Try to understand why the game was won or lost and identify moves which were weak and how the mistakes could have been avoided.  
Every game is played with the same seed, such that enemies and draws will be ordered the same in each game. Take advantage of this to memorize "random" draws across games and to figure out which moves work and don't work by trial and error.

Current text files:
{context}

Last game result JSON:
{json.dumps(result, indent=2)}
"""


def move_error_feedback(exc: Exception, response: str | None) -> str:
    response_text = (response or "").strip()
    reason = str(exc) or exc.__class__.__name__
    if response_text:
        return (
            f"The previous model response failed to parse or validate.\n"
            f"Failure reason: {reason}\n"
            f"Previous model response:\n{response_text}"
        )
    return (
        f"The previous move request failed before a model response could be used.\n"
        f"Failure reason: {reason}"
    )


def run_one(args: argparse.Namespace, ollama: Ollama, game_no: int, progress=print) -> dict:
    seed = game_seed(args, game_no)
    game = Game.new(seed=seed)
    illegal = 0
    last_error = ""
    memory = ""
    log_path = args.run_dir / f"{game_no-1:03d}_log.jsonl"
    progress(f"[game {game_no}] Starting game with seed={seed!r} (seed mode: {args.seed_mode})")
    progress(f"[game {game_no}] Writing detailed turn log to {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        turn = 0
        while game.phase not in {Phase.WON, Phase.LOST} and illegal < args.max_illegal:
            progress(f"[game {game_no} turn {turn + 1} enemy_pile {len(game.enemy_pile)}] Phase={game.phase.value}; illegal moves={illegal}/{args.max_illegal}")
            context = (args.run_dir / f"{game_no-1:03d}_strategy.txt").read_text(errors="replace")
            before = game.render()
            memory_before = memory
            comment = ""
            response = ""
            try:
                #progress(f"[game {game_no} turn {turn + 1} enemy_pile {len(game.enemy_pile)}] Requesting move from Ollama model {getattr(ollama, 'model', 'unknown')!r}")
                response = prompt_ollama(ollama, move_prompt(game, context, memory, illegal, last_error), progress=progress)
                slots, comment, memory = parse_agent_response(response)
                #progress(f"[game {game_no} turn {turn + 1} enemy_pile {len(game.enemy_pile)}] Model selected slot(s): {slots}")
                if game.phase == Phase.PLAY:
                    game.play_slots(slots)
                else:
                    game.discard_slots(slots)
                last_error = ""
                #progress(f"[game {game_no} turn {turn + 1} enemy_pile {len(game.enemy_pile)}] Move applied; new phase={game.phase.value}")
            except Exception as exc:  # keep games moving; illegal engine moves and ollama failures both count
                slots = []
                illegal += 1
                last_error = move_error_feedback(exc, response)
                progress(f"[game {game_no} turn {turn + 1} enemy_pile {len(game.enemy_pile)}] Illegal move or agent error ({illegal}/{args.max_illegal}): {last_error}")
            turn += 1
            log.write(json.dumps({
                "turn": turn,
                "phase_before": before,
                "response": response,
                "slots": slots,
                "comment": comment,
                "memory_before": memory_before,
                "memory_after": memory,
                "illegal_moves": illegal,
                "error": last_error,
                "phase_after": game.render(),
            }) + "\n")
    if illegal >= args.max_illegal and game.phase not in {Phase.WON, Phase.LOST}:
        game.phase = Phase.LOST
        game.message = f"Stopped after {illegal} illegal move(s)."
    progress(f"[game {game_no}] Finished with phase={game.phase.value}, score={score(game, illegal)}, illegal moves={illegal}")
    result = {
        "game": game_no,
        "seed": seed,
        "seed_mode": args.seed_mode,
        "phase": game.phase.value,
        "score": score(game, illegal),
        "illegal_moves": illegal,
        "log": str(log_path),
        "output_dir": str(args.run_dir),
    }
    print(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Regicide with a local Ollama LLM agent.")
    parser.add_argument("--model", default="llama3", help="Ollama model name.")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--context-dir", type=Path, default=Path("agent_context"))
    parser.add_argument("--log-dir", type=Path, default=Path("agent_logs"))
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base random seed. With --seed-mode fixed, every game uses this exact seed. "
            "With increment, game N uses seed + N. Omit for Python RNG randomness."
        ),
    )
    parser.add_argument(
        "--seed-mode",
        choices=("fixed", "increment", "random"),
        default="increment",
        help=(
            "How to seed multiple games: fixed reuses --seed, increment uses --seed + game number, "
            "random draws a fresh seed for each game."
        ),
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-illegal", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300, help="Seconds to wait for one Ollama /api/generate response.")
    parser.add_argument("--retries", type=int, default=0, help="Retries after transport, timeout, or JSON errors. Illegal moves are not retried.")
    parser.add_argument("--num-predict", type=int, default=25000, help="Ollama num_predict option; keep small because only three short lines are needed.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Ollama temperature option for move generation.")
    parser.add_argument("--revise-between", action="store_true", help="After each game, ask Ollama to rewrite context-dir/strategy.txt.")
    parser.add_argument("--stream", type=bool, default=False, help="More verbose printed outputs for debugging purposes.")
    parser.add_argument(
        "--check-ollama",
        action="store_true",
        help="Only test Ollama server communication, printing each progress step, then exit.",
    )
    args = parser.parse_args()
    ollama = Ollama(args.model, args.ollama_url, args.timeout, args.retries, args.num_predict, args.temperature, args.stream)
    if args.check_ollama:
        print(json.dumps(ollama.check_connection(), indent=2))
        return
    args.run_dir = create_run_dir(args.log_dir)
    shutil.copy2(args.context_dir / "strategy.txt", args.run_dir / "000_strategy.txt")

    for game_no in range(1, args.games + 1):
        result = run_one(args, ollama, game_no)
        if args.revise_between:
            print(f"[game {game_no}] Requesting revised strategy notes from Ollama model {getattr(ollama, 'model', 'unknown')!r}")
            ollama.num_predict *= 10; ollama.timeout *= 10
            current=(args.run_dir / f"{game_no-1:03d}_strategy.txt").read_text(errors="replace")
            text = prompt_ollama(ollama, revise_prompt(current, result), progress=print)
            (args.run_dir / f"{game_no:03d}_strategy.txt").write_text(text)
            print(f"[game {game_no}] Wrote revised strategy notes")
            ollama.num_predict /= 10; ollama.timeout /= 10


if __name__ == "__main__":
    main()
