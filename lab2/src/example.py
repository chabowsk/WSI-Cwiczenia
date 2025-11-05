from two_player_games.games.morris import SixMensMorris # or any other game
from two_player_games.games.dots_and_boxes import DotsAndBoxes
import random


game = DotsAndBoxes()

while not game.is_finished():
    moves = game.get_moves()
    move = random.choice(moves)
    game.make_move(move)

winner = game.get_winner()
if winner is None:
    print('Draw!')
else:
    print('Winner: Player ' + winner.char)
