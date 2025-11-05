import unittest

from two_player_games.games.nim import Nim, NimMove


class TestNim(unittest.TestCase):
    def test_init(self):
        game = Nim((3, 4, 5))
        self.assertEqual(str(game), f'Current player: {Nim.FIRST_PLAYER_DEFAULT_CHAR}\n1: |||\n2: ||||\n3: |||||')

    def test_get_moves(self):
        game = Nim((2, 0, 1))
        self.assertListEqual(game.get_moves(), [NimMove(0, 1), NimMove(0, 2), NimMove(2, 1)])

    def test_moves(self):
        game = Nim((3, 4, 5))

        game.make_move(NimMove(0, 2))
        self.assertIs(game.get_current_player(), game.second_player)
        self.assertFalse(game.is_finished())
        self.assertEqual(str(game), f'Current player: {Nim.SECOND_PLAYER_DEFAULT_CHAR}\n1: |\n2: ||||\n3: |||||')

        game.make_move(NimMove(2, 5))
        self.assertIs(game.get_current_player(), game.first_player)
        self.assertFalse(game.is_finished())
        self.assertEqual(str(game), f'Current player: {Nim.FIRST_PLAYER_DEFAULT_CHAR}\n1: |\n2: ||||\n3: ')

        game.make_move(NimMove(0, 1))
        self.assertIs(game.get_current_player(), game.second_player)
        self.assertFalse(game.is_finished())
        self.assertEqual(str(game), f'Current player: {Nim.SECOND_PLAYER_DEFAULT_CHAR}\n1: \n2: ||||\n3: ')

        game.make_move(NimMove(1, 4))
        self.assertTrue(game.is_finished())
        self.assertEqual(str(game), f'Winner: {Nim.FIRST_PLAYER_DEFAULT_CHAR}\n1: \n2: \n3: ')