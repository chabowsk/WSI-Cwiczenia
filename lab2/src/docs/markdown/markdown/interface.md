# Two-player game interface

Two primary base classes of the games are Game and State.

These two classes have very similar functionalities, with the following differences:

* a Game object contains a State object
* making move on a Game object changes its state while making move on a State object returns new state without affecting the previous one.
* a State object may have more game-specific functionalities

Common functionalities of Game and State:

* getting the list of avaliable moves
* getting current player
* checking if a game has finished

## Usage example

```python
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
```

## Intelligent move selection example

```python
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
```

## Interface classes

### *class* two_player_games.Game(state: [State](#two_player_games.State))

Game interface.

#### \_\_str_\_() → str

Returns string representation of the current game’s state.

* **Returns:**
  printable text represenation of the game’s state.
* **Return type:**
  str

#### get_current_player() → [Player](#two_player_games.Player)

Returns the current player.

* **Returns:**
  the object that represents the current player.
* **Return type:**
  [Player](#two_player_games.Player)

#### get_moves() → Iterable[[Move](#two_player_games.Move)]

Returns possible moves in the current state.

* **Returns:**
  An iterable of game-specific Move objects
  : that are valid for the current player in the current state.
* **Return type:**
  Iterable[[Move](#two_player_games.Move)]

#### get_players() → Iterable[[Player](#two_player_games.Player)]

Retrieves players. Their order may not be consistent between different states.

* **Returns:**
  the players in the game.
* **Return type:**
  Iterable[[Player](#two_player_games.Player)]

#### get_winner() → [Player](#two_player_games.Player) | None

Checks which player is the winner.

* **Returns:**
  Player object that represents the winner or None if not finished or draw.
* **Return type:**
  Optional[[Player](#two_player_games.Player)]

#### is_finished() → bool

Checks if the game is finished.

* **Returns:**
  if the game is finished.
* **Return type:**
  bool

#### make_move(move: [Move](#two_player_games.Move))

Makes move and changes the underlying state of the game.

* **Parameters:**
  **move** ([*Move*](#two_player_games.Move)) – the move to make.

### *class* two_player_games.Move

A base class for classes that represent moves in games.

### *class* two_player_games.Player(char: str)

A class that represents a player in a game

#### \_\_init_\_(char: str) → None

Initializes a player.

* **Parameters:**
  **char** – a single-character string to represent the player in textual representations of game state

### *class* two_player_games.State(current_player, other_player)

Immutable game state object.

#### \_\_str_\_() → str

Returns string representation of the current game’s state.

* **Returns:**
  printable text represenation of the game’s state.
* **Return type:**
  str

#### get_current_player() → [Player](#two_player_games.Player)

Returns the current player.

* **Returns:**
  the object that represents the current player.
* **Return type:**
  [Player](#two_player_games.Player)

#### get_moves() → Iterable[[Move](#two_player_games.Move)]

Returns possible moves in the current state.

* **Returns:**
  An iterable of game-specific Move objects
  : that are valid for the current player in the current state.
* **Return type:**
  Iterable[[Move](#two_player_games.Move)]

#### get_players() → Iterable[[Player](#two_player_games.Player)]

Retrieves players. Their order may not be consistent between different states.

* **Returns:**
  the players in the game.
* **Return type:**
  Iterable[[Player](#two_player_games.Player)]

#### get_winner() → [Player](#two_player_games.Player) | None

Checks which player is the winner.

* **Returns:**
  Player object that represents the winner or None if not finished or draw.
* **Return type:**
  Optional[[Player](#two_player_games.Player)]

#### is_finished() → bool

Checks if the game is finished.

* **Returns:**
  if the game is finished.
* **Return type:**
  bool

#### make_move(move: [Move](#two_player_games.Move)) → [State](#two_player_games.State)

Makes move without changing this object - returns a new object with given state.

* **Parameters:**
  **move** ([*Move*](#two_player_games.Move)) – the move to make.
* **Returns:**
  New state after the move
* **Return type:**
  [State](#two_player_games.State)
