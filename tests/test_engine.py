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


def test_exact_defeat_places_enemy_on_draw_pile():
    game = Game.new(seed=1)
    enemy = Card(Rank.JACK, Suit.CLUBS)
    game.active_enemy = enemy
    game.enemy_pile = []
    game.draw_pile = []
    game.hand = [Card(Rank.TEN, Suit.CLUBS)] + [None] * 7
    game.play_slots([1])
    assert game.phase == Phase.WON
    assert enemy in game.hand


def test_discard_requires_enough_value():
    game = Game.new(seed=1)
    game.active_enemy = Card(Rank.JACK, Suit.CLUBS)
    game.phase = Phase.DISCARD
    game.hand = [Card(Rank.TWO, Suit.HEARTS)] + [None] * 7
    with pytest.raises(ValueError, match="at least 10"):
        game.discard_slots([1])
