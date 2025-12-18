from two_player_games.state import State
from two_player_games.player import Player
from two_player_games.games.dots_and_boxes import DotsAndBoxesState, DotsAndBoxesMove
import math
import random

class Solver:
    def heuristic(self, state: State, maximizingPlayer: Player) -> float:       
        if state.is_finished():
            winner = state.get_winner()
            if winner is None:
                return 0.0
            elif winner.char == maximizingPlayer.char:
                return 1000.0
            else:
                return -1000.0
        
        scores = state.get_scores()
        maxPlayer_score = 0
        minPlayer_score = 0
        
        for player, score in scores.items():
            if player.char == maximizingPlayer.char:
                maxPlayer_score += score
            else:
                minPlayer_score += score
        
        return float(maxPlayer_score - minPlayer_score)
    
    def order_moves(self, state: State, moves: list) -> list:
        """
        HEURYSTYKA KOLEJNOŚCI RUCHÓW (Move Ordering):
        Priorytetyzuje ruchy, które zdobywają punkt (zamykają pudełko).
        """
        if not isinstance(state, DotsAndBoxesState):
            random.shuffle(moves)
            return moves

        scoring_moves = []
        normal_moves = []

        for move in moves:
            next_state = state.make_move(move)
            if next_state.get_current_player().char == state.get_current_player().char:
                scoring_moves.append(move)
            else:
                normal_moves.append(move)

        random.shuffle(scoring_moves)
        random.shuffle(normal_moves)

        # Najpierw ruchy punktujące, potem reszta
        return scoring_moves + normal_moves

    def alphabeta(self, state: State, depth: int, a: float, b: float, maximizingPlayer: Player) -> float:
        if depth == 0 or state.is_finished():
            return self.heuristic(state, maximizingPlayer)
        
        moves = state.get_moves()

        moves = self.order_moves(state, moves)

        current_player_char = state.get_current_player().char
        
        if current_player_char == maximizingPlayer.char:
            value = -math.inf
            for move in moves:
                value = max(value, self.alphabeta(state.make_move(move), depth-1, a, b, maximizingPlayer))
                a = max(a, value)
                if value >= b:
                    break
            return value
        else:
            value = math.inf
            for move in moves:
                child = state.make_move(move)
                value = min(value, self.alphabeta(child, depth-1, a, b, maximizingPlayer))
                b = min(b, value)
                if value <= a:
                    break
            return value