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
