# Regicide Solo

A small Python implementation of the solo mode for the card game Regicide, with a simple text CLI intended to be easy for humans or text-only agents to drive.

## Run

```bash
python -m regicide.cli --seed 1
```

or, after installing the package:

```bash
regicide-solo --seed 1
```

On each turn, enter one-based hand slot numbers separated by spaces. During the play phase those slots are played as an attack. During the discard phase those slots are discarded to absorb the enemy attack.

## Rule coverage

This engine intentionally keeps the implementation compact, but it includes the core solo flow:

- a 40-card tavern draw pile made from A-10 in all suits;
- a 12-card enemy pile of jacks, queens, and kings;
- an eight-card solo hand;
- legal play validation for singles, same-rank sets, and ace companions up to value 10;
- suit effects for clubs, diamonds, hearts, and spades;
- exact enemy defeats returning the defeated enemy to the tavern deck;
- play and discard phases with text validation errors.

The CLI is deliberately plain text. For example, the first screen from `python -m regicide.cli --seed 1` looks like:

```text
Enter slot numbers separated by spaces. Type 'quit' to exit.

Draw pile (32)
Discard pile (0)
Enemy pile (11)
Active enemy: J♥
Enemy damage: 0/20  Incoming attack: 10
In play: --
Hand:
 1    2    3    4    5    6    7    8
9♣   7♠   5♣   7♦   8♣   2♠   9♥   A♠
Phase: Play
Play slots>
```

## Ollama learning agent runner

This repo also includes a deliberately small agent loop for a local Ollama server. It starts a full game, asks Ollama for slot numbers, applies the move to the engine, sends the next rendered state back to Ollama, and writes a JSONL session log containing every state, response, parsed move, and illegal-move error.

```bash
python -m regicide.agent --model llama3 --seed 1
```

The installed console script is equivalent:

```bash
regicide-agent --model llama3 --seed 1
```

By default the runner:

- reads all `*.txt` files from `agent_context/` and includes them in every move prompt;
- writes logs to `agent_logs/`;
- talks to `http://localhost:11434/api/generate` with `stream: false`;
- uses a 20 second timeout, two retries, then gives up for that prompt;
- counts illegal engine moves and Ollama communication failures as illegal moves;
- stops early after 10 illegal moves;
- prints one JSON result per game with the terminal phase, illegal-move count, score, and log path.

The final score is:

```text
10000 * enemies defeated
+ 100 * sum(value of cards in hand)
+ cards in the draw pile
- 1000 * illegal moves attempted
```

To run repeated games and let the model revise its local context between games, use `--games` with `--revise-between`. After each game, the runner asks Ollama to rewrite `agent_context/strategy.txt` from the current text files and the previous game result. Each game gets its own output folder under `--log-dir`, containing `game.jsonl` plus a `context/` snapshot of the `.txt` files used for that game.

```bash
python -m regicide.agent --model llama3 --games 25 --revise-between
```

Seed behavior is explicit for multi-game runs. Use `--seed-mode fixed --seed 1` to replay the same starting game repeatedly while the strategy evolves, `--seed-mode random` to draw a fresh seed for every game, or the default `--seed-mode increment --seed 1` to use `seed + game_number`.

Useful knobs:

```bash
python -m regicide.agent \
  --model llama3 \
  --ollama-url http://localhost:11434 \
  --context-dir agent_context \
  --log-dir agent_logs \
  --seed 1 \
  --seed-mode fixed \
  --timeout 10 \
  --retries 1 \
  --max-illegal 10
```

The prompting and context format are intentionally plain. Edit the local text files and/or `regicide/agent.py` if you want a richer policy, move format, or self-improvement protocol.
