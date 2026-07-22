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
    comment = limit_words("\n".join(part for part in sections["comment"] if part), max_words = 1000)
    memory = limit_words("\n".join(part for part in sections["memory"] if part), max_words = 10000)
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
                        if progress is not None and now - last_flush > 5.0:
                    
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

def get_text(directory: Path, filename: str) -> str:
    return (directory / filename).read_text(encoding="utf-8", errors="replace")

def get_text_all_previous(directory: Path, tail_filename: str, game_no: int) -> str:
    parts = []

    for i in range(game_no):
        filename = f"{i:03d}_{tail_filename}"
        parts.append(
            f"Summary game {i + 1}:\n"
            f"{get_text(directory, filename).strip()}"
        )
    return "\n\n".join(parts)

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


def move_prompt(game: Game, game_no: int, memory: str, illegal_moves: int, last_error: str, run_dir: Path) -> str:
    return f"""You are an LLM agent playing solo Regicide. The play and discard phases of each turn are processed as separate prompts. Reply only in this exact three-line format:
1: <space seperated hand slot indices>
2: <optional brief comment, up to 1000 words, explaining the choice in the game log>
3: <optional short-term memory, up to 10000 words, repeated back to the agent in next phase prompt>

For example, this is a good attempt at a valid response:
1: 1 7
2: Play spades against undamaged enemy to maximize value
3: Nothing important to remember
This is a correctly formatted single card play or single card discard:
1: 5
This is a correctly formatted 0-card discard. 0-card play is not allowed.
1: 

This is the current game state:
{game.render()}

Short-term memory from previous turn performed by the same agent:
{memory or '(none)'}

Use the rules and strategies to determine valid and good plays. Note that the game progresses predictably through play and discard phases, allowing planning ahead for future turns. New information is only gained when cards are revealed from the Enemy pile or Draw pile, and even then only if they are not already known.
The comment is logged for later strategy revision but is not shown to future move prompts. The comment should help explain the agent's decision, ideally connecting it with the contents of strategy.txt
The memory is repeated back to the agent in the next phase prompt: in light of your own thinking, consider what reminders, rules clarifications and similar advice would be useful to carry forward. In particular, clear up misunderstandings that wasted a lot of the thinking budget. Use it also to retain information about the game state that becomes hidden, such as which cards of which suit are left in the enemy pile and known cards on top of the draw pile.

Rules for the game (these are authorative):
{get_text(run_dir, "rules.txt")}

Strategic advice for how to play (these are based on previous experiences of the LLM agent):
{get_text(run_dir, f"{game_no-1:03d}_strategy.txt")}

Summaries of previous games (these are written by the LLM agent):
{get_text_all_previous(run_dir, "summary.txt", game_no)}

Illegal moves so far: {illegal_moves}
Last move error feedback (use this to fix your next response): {last_error or 'none'}

Repeating the key task:
Reply only in this exact three-line format:
1: <space seperated hand slot indices>
2: <optional brief comment, up to 1000 words, explaining the choice in the game log>
3: <short-term memory, up to 10000 words, carried over to the next turn of the game>

Repeating the current game state:
{game.render()}
"""


def revise_prompt(game_no: int, run_dir: Path) -> str:
    return f"""You are an LLM agent playing solo Regicide. You are not playing the actual game, currently, but instead revising a document explaining how to play the game well, aimed at helping an LLM agent make good decisions during a game.
You are provided with the previous version of this document. Revise it and return only the complete, new version of the document.
Make sure to retain the useful parts of the old document, while trying to improve it.
The main success criteria when playing is to avoid illegal moves and defeat more enemies before losing. While it is difficult, it is possible to defeat all 12 enemies in a single game with strong play.
Note that every game is played with the same seed. This means that the starting hand, the Draw pile and the Enemy pile always start out in the same state, including the order of the cards. Try to take advantage of this to improve play from game to game.

Log file from previous game:
{get_text(run_dir, f"{game_no-1:03d}_log.jsonl")}

Summaries of previous games (these are written by the LLM agent):
{get_text_all_previous(run_dir, "summary.txt", game_no)}

Previous version of strategy document:
{get_text(run_dir, f"{game_no-1:03d}_strategy.txt")}

Repeating the key task:
You are an LLM agent playing solo Regicide. You are not playing the actual game, currently, but instead revising a document explaining how to play the game well, aimed at helping an LLM agent make good decisions during a game.
You are provided with the previous version of this document. Revise it and return only the complete, new version of the document.
Make sure to retain the useful parts of the old document, while trying to improve it.
"""

def summarize_prompt(game_no: int, run_dir: Path) -> str:
    return f"""You are an LLM agent playing solo Regicide. You are not playing the actual game, currently, but instead summarizing the game you just played.
The main success criteria when playing is to avoid illegal moves and defeat more enemies before losing. While it is difficult, it is possible to defeat all 12 enemies in a single game with strong play.
Note that every game is played with the same seed. This means that the starting hand, the Draw pile and the Enemy pile always start out in the same state, including the order of the cards. Try to take advantage of this to improve play from game to game.
Write a concise summary of what happened during the game, on a turn-by-turn basis if useful. Avoiding duplicating information and bloating the summary. Write your analysis of which moves were good or bad and why. Try to determine how the game ended and why. Reflect on whether the game was played according to the advice in the stategy document and whether the advice was useful.

Log file from previous game:
{get_text(run_dir, f"{game_no-1:03d}_log.jsonl")}

Summaries of previous games (these are written by the LLM agent):
{get_text_all_previous(run_dir, "summary.txt", game_no)}

Previous version of strategy document:
{get_text(run_dir, f"{game_no-1:03d}_strategy.txt")}

Repeating the key task:
You are an LLM agent playing solo Regicide. You are not playing the actual game, currently, but instead summarizing the game you just played.
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
            before = game.render()
            memory_before = memory
            print(before)
            comment = ""
            response = ""
            try:
                progress(f"[game {game_no} turn {turn + 1}] Requesting move from Ollama model {getattr(ollama, 'model', 'unknown')!r}")
                response = prompt_ollama(ollama, move_prompt(game, game_no, memory, illegal, last_error, args.run_dir), progress=progress)
                slots, comment, memory = parse_agent_response(response)
                if game.phase == Phase.PLAY:
                    game.play_slots(slots)
                else:
                    game.discard_slots(slots)
                last_error = ""
                print(response)
            except Exception as exc:  # keep games moving; illegal engine moves and ollama failures both count
                slots = []
                illegal += 1
                last_error = move_error_feedback(exc, response)
                progress(f"[game {game_no} turn {turn + 1} enemy_pile {len(game.enemy_pile)}] Illegal move or agent error ({illegal}/{args.max_illegal}): {last_error}")
            turn += 1
            log.write(json.dumps({
                "turn": turn,
                "game_state_before": before,
                "response": response,
                "slots": slots,
                "comment": comment,
                "memory_before": memory_before,
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
            "memory": memory,
            "score": score(game, illegal),
            "illegal_moves": illegal,
            "log": str(log_path),
            "output_dir": str(args.run_dir),
        }
        for key,val in result.items(): print(f"{key}: {val}")
        log.write(json.dumps(result) + "\n")
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
    parser.add_argument("--think", type=bool, default=True, help="Enable thinking for LLM prompts")
    parser.add_argument("--stream", type=bool, default=False, help="More verbose printed outputs for debugging purposes.")
    parser.add_argument(
        "--check-ollama",
        action="store_true",
        help="Only test Ollama server communication, printing each progress step, then exit.",
    )
    args = parser.parse_args()
    ollama = Ollama(args.model, args.ollama_url, args.timeout, args.retries, args.num_predict, args.temperature, args.think, args.stream)
    if args.check_ollama:
        print(json.dumps(ollama.check_connection(), indent=2))
        return
    args.run_dir = create_run_dir(args.log_dir)
    shutil.copy2(args.context_dir / "strategy.txt", args.run_dir / "000_strategy.txt")
    shutil.copy2(args.context_dir / "rules.txt", args.run_dir / "rules.txt")

    for game_no in range(1, args.games + 1):
        result = run_one(args, ollama, game_no)
        
        ollama.num_predict *= 10; ollama.timeout *= 10
        print(f"[game {game_no}] Requesting game summary from Ollama model {getattr(ollama, 'model', 'unknown')!r}")
        text = prompt_ollama(ollama, summarize_prompt(game_no, args.run_dir), progress=print)        
        (args.run_dir / f"{game_no:03d}_summary.txt").write_text(text)
        
        print(f"[game {game_no}] Requesting revised strategy notes from Ollama model {getattr(ollama, 'model', 'unknown')!r}")
        text = prompt_ollama(ollama, revise_prompt(current, result), progress=print)
        (args.run_dir / f"{game_no:03d}_strategy.txt").write_text(text)
        
        ollama.num_predict /= 10; ollama.timeout /= 10


if __name__ == "__main__":
    main()
