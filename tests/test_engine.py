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
