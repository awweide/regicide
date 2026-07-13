from pathlib import Path

import pytest

from regicide.engine import Card, Game, Phase, Rank, Suit


def test_new_game_has_solo_hand_and_enemy():
    game = Game.new(seed=1)
    assert len([card for card in game.hand if card is not None]) == 8
    assert game.active_enemy is not None
    assert game.phase == Phase.PLAY


def test_rejects_mismatched_non_ace_multi_play():
    game = Game.new(seed=1)
    game.hand = [Card(Rank.TWO, Suit.CLUBS), Card(Rank.THREE, Suit.CLUBS)] + [None] * 6
    with pytest.raises(ValueError, match="same rank"):
        game.play_slots([1, 2])


def test_exact_defeat_places_enemy_on_draw_pile_without_drawing():
    game = Game.new(seed=1)
    enemy = Card(Rank.JACK, Suit.HEARTS)
    game.active_enemy = enemy
    game.enemy_pile = []
    game.draw_pile = []
    game.hand = [Card(Rank.TEN, Suit.CLUBS)] + [None] * 7
    game.play_slots([1])
    assert game.phase == Phase.WON
    assert game.hand == [None] * 8
    assert game.draw_pile == [enemy]


def test_defeating_enemy_reveals_next_enemy_in_play_phase():
    game = Game.new(seed=1)
    next_enemy = Card(Rank.JACK, Suit.DIAMONDS)
    game.active_enemy = Card(Rank.JACK, Suit.HEARTS)
    game.enemy_pile = [next_enemy]
    game.hand = [Card(Rank.TEN, Suit.CLUBS)] + [Card(Rank.TEN, Suit.SPADES)] * 7
    game.play_slots([1])
    assert game.active_enemy == next_enemy
    assert game.phase == Phase.PLAY


def test_multi_suit_play_uses_total_value_for_each_suit_power():
    game = Game.new(seed=1)
    game.active_enemy = Card(Rank.JACK, Suit.HEARTS)
    game.hand = [Card(Rank.ACE, Suit.SPADES), Card(Rank.EIGHT, Suit.CLUBS)] + [None] * 6
    game.play_slots([1, 2])
    assert game.enemy_damage == 18
    assert game.attack_reduction == 9
    assert game.incoming_attack == 1


def test_enemy_suit_blocks_only_matching_suit_power_from_multi_suit_play():
    game = Game.new(seed=1)
    game.active_enemy = Card(Rank.JACK, Suit.CLUBS)
    game.hand = [Card(Rank.ACE, Suit.SPADES), Card(Rank.EIGHT, Suit.CLUBS)] + [None] * 6
    game.play_slots([1, 2])
    assert game.enemy_damage == 9
    assert game.attack_reduction == 9
    assert game.incoming_attack == 1


def test_play_compacts_hand_after_removing_played_cards():
    game = Game.new(seed=1)
    game.active_enemy = Card(Rank.JACK, Suit.HEARTS)
    game.hand = [
        Card(Rank.TWO, Suit.CLUBS),
        Card(Rank.THREE, Suit.SPADES),
        Card(Rank.FOUR, Suit.DIAMONDS),
        Card(Rank.FIVE, Suit.HEARTS),
        None,
        None,
        None,
        None,
    ]
    game.play_slots([2])
    assert game.hand == [
        Card(Rank.TWO, Suit.CLUBS),
        Card(Rank.FOUR, Suit.DIAMONDS),
        Card(Rank.FIVE, Suit.HEARTS),
        None,
        None,
        None,
        None,
        None,
    ]


def test_render_aligns_slot_numbers_with_cards():
    game = Game.new(seed=1)
    game.hand = [Card(Rank.ACE, Suit.SPADES), Card(Rank.TEN, Suit.CLUBS)] + [None] * 6
    lines = game.render().splitlines()
    hand_index = lines.index("Hand:")
    assert lines[hand_index + 1] == " 1     2    3    4    5    6    7    8"
    assert lines[hand_index + 2] == "A♠   10♣   --   --   --   --   --   --"


def test_discard_requires_enough_value():
    game = Game.new(seed=1)
    game.active_enemy = Card(Rank.JACK, Suit.CLUBS)
    game.phase = Phase.DISCARD
    game.hand = [Card(Rank.TWO, Suit.HEARTS)] + [None] * 7
    with pytest.raises(ValueError, match="at least 10"):
        game.discard_slots([1])


def test_agent_score_counts_defeated_enemies_hand_draw_and_illegal_moves():
    from regicide.agent import score

    game = Game.new(seed=1)
    game.enemy_pile = game.enemy_pile[:10]
    game.active_enemy = Card(Rank.JACK, Suit.CLUBS)
    game.hand = [Card(Rank.TEN, Suit.HEARTS), Card(Rank.ACE, Suit.SPADES)] + [None] * 6
    game.draw_pile = [Card(Rank.TWO, Suit.CLUBS)] * 3
    assert score(game, illegal_moves=2) == 10000 + 1100 + 3 - 2000


def test_agent_seed_modes_for_multiple_games():
    from argparse import Namespace

    from regicide.agent import game_seed

    assert game_seed(Namespace(seed=7, seed_mode="fixed"), 1) == 7
    assert game_seed(Namespace(seed=7, seed_mode="fixed"), 2) == 7
    assert game_seed(Namespace(seed=7, seed_mode="increment"), 1) == 8
    assert game_seed(Namespace(seed=7, seed_mode="increment"), 2) == 9
    assert game_seed(Namespace(seed=None, seed_mode="increment"), 1) is None


def test_agent_run_one_writes_per_game_log_and_context_snapshot(tmp_path):
    from argparse import Namespace

    from regicide.agent import Ollama, run_one

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "strategy.txt").write_text("play clubs first")
    args = Namespace(
        seed=1,
        seed_mode="fixed",
        log_dir=tmp_path / "logs",
        context_dir=context_dir,
        max_illegal=0,
    )
    result = run_one(args, Ollama("unused"), 1)

    output_dir = Path(result["output_dir"])
    if not output_dir.is_absolute():
        output_dir = tmp_path / output_dir
    assert result["seed"] == 1
    assert result["seed_mode"] == "fixed"
    assert Path(result["log"]).name == "game.jsonl"
    assert Path(result["log"]).parent == output_dir
    assert (output_dir / "game.jsonl").exists()
    assert (output_dir / "context" / "strategy.txt").read_text() == "play clubs first"


def test_agent_parses_structured_move_comment_and_memory():
    from regicide.agent import parse_agent_response

    slots, comment, memory = parse_agent_response(
        "1: 1 3\n"
        "2: Play a diamond pair to refill hand.\n"
        "3: J♣ is on top of the draw pile.\n"
    )

    assert slots == [1, 3]
    assert comment == "Play a diamond pair to refill hand."
    assert memory == "J♣ is on top of the draw pile."


def test_agent_response_limits_comment_and_memory_to_200_words():
    from regicide.agent import parse_agent_response

    long_text = " ".join(f"word{i}" for i in range(205))
    _, comment, memory = parse_agent_response(f"1: 1\n2: {long_text}\n3: {long_text}")

    assert len(comment.split()) == 200
    assert len(memory.split()) == 200


def test_agent_run_one_logs_comment_and_short_term_memory(tmp_path):
    from argparse import Namespace
    import json

    from regicide.agent import run_one

    class FakeOllama:
        def prompt(self, prompt: str) -> str:
            assert "Short-term memory from previous turn:" in prompt
            return "1: 1 1\n2: intentionally duplicate move for parser coverage\n3: remember top card is J♣"

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    args = Namespace(
        seed=1,
        seed_mode="fixed",
        log_dir=tmp_path / "logs",
        context_dir=context_dir,
        max_illegal=1,
    )

    result = run_one(args, FakeOllama(), 1)
    entry = json.loads(Path(result["log"]).read_text().splitlines()[0])

    assert entry["comment"] == "intentionally duplicate move for parser coverage"
    assert entry["memory_before"] == ""
    assert entry["memory_after"] == "remember top card is J♣"


def test_agent_run_one_reports_progress_updates(tmp_path):
    from argparse import Namespace

    from regicide.agent import run_one

    class FakeOllama:
        model = "fake-model"

        def prompt(self, prompt: str) -> str:
            return "1: 1 1\n2: duplicate move\n3: remembered state"

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    progress_messages: list[str] = []
    args = Namespace(
        seed=1,
        seed_mode="fixed",
        log_dir=tmp_path / "logs",
        context_dir=context_dir,
        max_illegal=1,
    )

    run_one(args, FakeOllama(), 1, progress=progress_messages.append)

    assert progress_messages[0].startswith("[game 1] Starting game")
    assert any("Requesting move from Ollama model 'fake-model'" in message for message in progress_messages)
    assert any("Illegal move or agent error (1/1)" in message for message in progress_messages)
    assert progress_messages[-1].startswith("[game 1] Finished")
