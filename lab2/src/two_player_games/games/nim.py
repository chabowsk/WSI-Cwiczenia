from typing import Dict, Iterable, List, Optional, Tuple
from two_player_games.game import Game
from two_player_games.move import Move
from two_player_games.player import Player
from two_player_games.state import State


class Nim(Game):
    """Class that represents the nim game (misere variant)"""
    FIRST_PLAYER_DEFAULT_CHAR = '1'
    SECOND_PLAYER_DEFAULT_CHAR = '2'

    def __init__(self, heaps: Iterable[int] = (7, 7, 7), first_player: Player = None, second_player: Player = None):
        """
        Initializes game.

        Parameters:
            size: the size of the game as number of columns and rows of boxes
            first_player: the player that will go first (if None is passed, a player will be created)
            second_player: the player that will go second (if None is passed, a player will be created)
        """
        self.first_player = first_player or Player(self.FIRST_PLAYER_DEFAULT_CHAR)
        self.second_player = second_player or Player(self.SECOND_PLAYER_DEFAULT_CHAR)

        state = NimState(self.first_player, self.second_player, heaps)

        super().__init__(state)


class NimMove(Move):
    """
    Class that represents a move in the nim game

    Variables:
        heap: int, heap number from which the elements are taken
        n: number of elements to take
    """
    def __init__(self, heap: int, n: int):
        self.heap = heap
        self.n = n

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, NimMove):
            return False
        return self.heap == value.heap and self.n == value.n


class NimState(State):
    """Class that represents a state in the nim game (misere variant)"""
    def __init__(self,
            current_player: Player, other_player: Player, heaps: Iterable[int]):
        """Creates the state. Do not call directly."""

        self.heaps = tuple(heaps)
        assert self.heaps and all(heap >= 0 for heap in self.heaps)

        super().__init__(current_player, other_player)

    def get_moves(self) -> Iterable[NimMove]:
        return [NimMove(i, n) for i, heap in enumerate(self.heaps) for n in range(1, heap + 1)]

    def make_move(self, move: NimMove) -> 'NimState':
        assert move.n > 0 and self.heaps[move.heap] >= move.n
        heaps = tuple(heap - move.n if i == move.heap else heap for i, heap in enumerate(self.heaps))
        return NimState(self._other_player, self._current_player, heaps)

    def is_finished(self) -> bool:
        return all(heap == 0 for heap in self.heaps)

    def get_winner(self) -> Optional[Player]:
        if not self.is_finished():
            return None
        else:
            return self._current_player  # previous player lost, because it picked the last elements

    def __str__(self) -> str:
        text = [('Winner: ' if self.is_finished() else 'Current player: ') + self._current_player.char]

        for i, heap in enumerate(self.heaps):
            text.append(str(i + 1) + ': ' + '|' * heap)

        return '\n'.join(text)
