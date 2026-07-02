from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Iterable


class Suit(str, Enum):
    CLUBS = "♣"
    DIAMONDS = "♦"
    HEARTS = "♥"
    SPADES = "♠"


class Rank(str, Enum):
    ACE = "A"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"


NORMAL_RANKS = [Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN]
ENEMY_RANKS = [Rank.JACK, Rank.QUEEN, Rank.KING]
CARD_VALUES = {
    Rank.ACE: 1,
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 10,
    Rank.QUEEN: 15,
    Rank.KING: 20,
}
ENEMY_HEALTH = {Rank.JACK: 20, Rank.QUEEN: 30, Rank.KING: 40}
ENEMY_ATTACK = {Rank.JACK: 10, Rank.QUEEN: 15, Rank.KING: 20}
HAND_SIZE_SOLO = 8


class Phase(str, Enum):
    PLAY = "Play"
    DISCARD = "Discard"
    WON = "Won"
    LOST = "Lost"


@dataclass(frozen=True, order=True)
class Card:
    rank: Rank
    suit: Suit

    @property
    def value(self) -> int:
        return CARD_VALUES[self.rank]

    @property
    def is_enemy(self) -> bool:
        return self.rank in ENEMY_RANKS

    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value}"


@dataclass
class PlayedSet:
    cards: list[Card]
    damage: int

    def __str__(self) -> str:
        return "(" + " + ".join(str(card) for card in self.cards) + ")"


@dataclass
class Game:
    draw_pile: list[Card] = field(default_factory=list)
    discard_pile: list[Card] = field(default_factory=list)
    enemy_pile: list[Card] = field(default_factory=list)
    hand: list[Card | None] = field(default_factory=lambda: [None] * HAND_SIZE_SOLO)
    active_enemy: Card | None = None
    in_play: list[PlayedSet] = field(default_factory=list)
    enemy_damage: int = 0
    attack_reduction: int = 0
    phase: Phase = Phase.PLAY
    message: str = ""

    @classmethod
    def new(cls, seed: int | None = None) -> "Game":
        rng = random.Random(seed)
        tavern = [Card(rank, suit) for suit in Suit for rank in NORMAL_RANKS]
        enemies_by_rank = [[Card(rank, suit) for suit in Suit] for rank in ENEMY_RANKS]
        rng.shuffle(tavern)
        enemy_pile: list[Card] = []
        for group in enemies_by_rank:  # Jacks first, then Queens, then Kings.
            rng.shuffle(group)
            enemy_pile.extend(group)
        game = cls(draw_pile=tavern, enemy_pile=enemy_pile)
        game._draw_to_hand_limit()
        game._reveal_next_enemy()
        return game

    def play_slots(self, slots: Iterable[int]) -> None:
        self._require_phase(Phase.PLAY)
        indexes = self._slot_indexes(slots)
        cards = [self.hand[index] for index in indexes]
        if any(card is None for card in cards):
            raise ValueError("Cannot play an empty hand slot.")
        play = list(cards)  # type: ignore[arg-type]
        self._validate_play(play)
        for index in indexes:
            self.hand[index] = None
        self._apply_play(play)
        if self.phase == Phase.PLAY and self.active_enemy is not None:
            if sum(card.value for card in self.hand if card is not None) < self.incoming_attack:
                self.phase = Phase.LOST
                self.message = "Enemy attack cannot be absorbed. You lose."
            else:
                self.phase = Phase.DISCARD
                self.message = "Enemy survived. Discard cards with enough value to absorb the attack."

    def discard_slots(self, slots: Iterable[int]) -> None:
        self._require_phase(Phase.DISCARD)
        indexes = self._slot_indexes(slots, allow_empty=True)
        cards = [self.hand[index] for index in indexes if self.hand[index] is not None]
        required = self.incoming_attack
        if sum(card.value for card in cards) < required:
            raise ValueError(f"Discard value must be at least {required}.")
        for index in indexes:
            card = self.hand[index]
            if card is not None:
                self.discard_pile.append(card)
                self.hand[index] = None
        self.phase = Phase.PLAY
        self.message = f"Discarded {len(cards)} card(s)."

    @property
    def incoming_attack(self) -> int:
        if self.active_enemy is None:
            return 0
        return max(0, ENEMY_ATTACK[self.active_enemy.rank] - self.attack_reduction)

    @property
    def active_enemy_health(self) -> int:
        if self.active_enemy is None:
            return 0
        return ENEMY_HEALTH[self.active_enemy.rank]

    def _apply_play(self, cards: list[Card]) -> None:
        damage = sum(card.value for card in cards)
        if any(card.suit == Suit.CLUBS for card in cards):
            damage *= 2
        self.enemy_damage += damage
        self.in_play.append(PlayedSet(cards, damage))
        if any(card.suit == Suit.SPADES for card in cards):
            self.attack_reduction += sum(card.value for card in cards if card.suit == Suit.SPADES)
        if any(card.suit == Suit.HEARTS for card in cards):
            self._heal(sum(card.value for card in cards if card.suit == Suit.HEARTS))
        if any(card.suit == Suit.DIAMONDS for card in cards):
            self._draw_cards(sum(card.value for card in cards if card.suit == Suit.DIAMONDS))
        if self.enemy_damage >= self.active_enemy_health:
            self._defeat_enemy(exact=self.enemy_damage == self.active_enemy_health)

    def _defeat_enemy(self, exact: bool) -> None:
        defeated = self.active_enemy
        assert defeated is not None
        if exact:
            self.draw_pile.insert(0, defeated)
        else:
            self.discard_pile.append(defeated)
        for played in self.in_play:
            self.discard_pile.extend(played.cards)
        self.in_play.clear()
        self.enemy_damage = 0
        self.attack_reduction = 0
        self.active_enemy = None
        self._draw_to_hand_limit()
        if not self._reveal_next_enemy():
            self.phase = Phase.WON
            self.message = "All enemies defeated. You win!"
        else:
            self.phase = Phase.PLAY
            self.message = f"Defeated {defeated}."

    def _validate_play(self, cards: list[Card]) -> None:
        if not cards:
            raise ValueError("Choose at least one card.")
        if any(card.is_enemy for card in cards) and len(cards) > 1:
            raise ValueError("Enemy cards in hand must be played alone.")
        if any(card.is_enemy for card in cards):
            return
        non_aces = [card for card in cards if card.rank != Rank.ACE]
        if len(non_aces) > 1:
            ranks = {card.rank for card in non_aces}
            if len(ranks) != 1:
                raise ValueError("Multiple non-ace cards must all have the same rank.")
        if sum(card.value for card in cards) > 10:
            raise ValueError("Played cards may not have total value greater than 10 before suit effects.")

    def _heal(self, amount: int) -> None:
        random.shuffle(self.discard_pile)
        healed = self.discard_pile[:amount]
        del self.discard_pile[:amount]
        self.draw_pile = healed + self.draw_pile

    def _draw_cards(self, amount: int) -> None:
        for _ in range(amount):
            if not self.draw_pile:
                return
            empty = self._first_empty_slot()
            if empty is None:
                return
            self.hand[empty] = self.draw_pile.pop()

    def _draw_to_hand_limit(self) -> None:
        while self.draw_pile and self._first_empty_slot() is not None:
            self.hand[self._first_empty_slot()] = self.draw_pile.pop()  # type: ignore[index]

    def _first_empty_slot(self) -> int | None:
        return next((index for index, card in enumerate(self.hand) if card is None), None)

    def _reveal_next_enemy(self) -> bool:
        if not self.enemy_pile:
            return False
        self.active_enemy = self.enemy_pile.pop(0)
        return True

    def _require_phase(self, phase: Phase) -> None:
        if self.phase != phase:
            raise ValueError(f"Expected phase {phase.value}, but current phase is {self.phase.value}.")

    def _slot_indexes(self, slots: Iterable[int], allow_empty: bool = False) -> list[int]:
        indexes = []
        for slot in slots:
            if slot < 1 or slot > len(self.hand):
                raise ValueError(f"Slot {slot} is outside 1-{len(self.hand)}.")
            index = slot - 1
            if not allow_empty and self.hand[index] is None:
                raise ValueError(f"Slot {slot} is empty.")
            indexes.append(index)
        if len(indexes) != len(set(indexes)):
            raise ValueError("Cannot choose the same slot more than once.")
        return indexes

    def render(self) -> str:
        enemy = str(self.active_enemy) if self.active_enemy else "--"
        in_play = ", ".join(str(played) for played in self.in_play) or "--"
        lines = [
            f"Draw pile ({len(self.draw_pile)})",
            f"Discard pile ({len(self.discard_pile)})",
            f"Enemy pile ({len(self.enemy_pile)})",
            f"Active enemy: {enemy}",
            f"Enemy damage: {self.enemy_damage}/{self.active_enemy_health}  Incoming attack: {self.incoming_attack}",
            f"In play: {in_play}",
            "Hand:",
            "   ".join(str(i) for i in range(1, len(self.hand) + 1)),
            "   ".join(str(card) if card is not None else "--" for card in self.hand),
            f"Phase: {self.phase.value}",
        ]
        if self.message:
            lines.append(f"Message: {self.message}")
        return "\n".join(lines)
