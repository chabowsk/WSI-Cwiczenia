Two-player game interface
=========================

Two primary base classes of the games are `Game` and `State`.

These two classes have very similar functionalities, with the following differences:

* a `Game` object contains a `State` object
* making move on a `Game` object changes its state while making move on a `State` object returns new state without affecting the previous one.
* a `State` object may have more game-specific functionalities

Common functionalities of `Game` and `State`:

* getting the list of avaliable moves
* getting current player
* checking if a game has finished

=============
Usage example
=============

.. code-block:: python

    from two_player_games.games.morris import SixMensMorris  # or any other game
    import random


    game = SixMensMorris()

    while not game.is_finished():
        moves = game.get_moves()
        move = random.choice(moves)
        game.make_move(move)

    winner = game.get_winner()
    if winner is None:
        print('Draw!')
    else:
        print('Winner: Player ' + winner.char)

==================================
Intelligent move selection example
==================================

.. code-block:: python

    from two_player_games.games.morris import SixMensMorris  # or any other game
    import random


    def score(current_player, state):
        # a game-dependent scoring function from the perspective of the current_player
        ...

    game = SixMensMorris()

    while not game.is_finished():
        moves = game.get_moves()
        
        move_scores = [
            (move, score(game.get_current_player(), game.state.make_move(move)))
            for move in moves
        ]  # note that `state.make_move()` does not change the game state

        move = max(move_scores, key=lambda ms: ms[1])[0]
        game.make_move(move)

    winner = game.get_winner()
    if winner is None:
        print('Draw!')
    else:
        print('Winner: Player ' + winner.char)

=================
Interface classes
=================

.. autoclass:: two_player_games.Game
    :members:
    :special-members: __str__

.. autoclass:: two_player_games.Move
    :members:

.. autoclass:: two_player_games.Player
    :members:
    :special-members: __init__

.. autoclass:: two_player_games.State
    :members:
    :special-members: __str__
