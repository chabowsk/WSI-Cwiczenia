import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from solver import Solver 

# Importy biblioteki gry z repozytorium: https://github.com/lychanl/two-player-games
try:
    from two_player_games.games.dots_and_boxes import DotsAndBoxes
    from two_player_games.move import Move
    from two_player_games.player import Player
except ImportError:
    print("Błąd: Nie znaleziono pakietu 'two_player_games'. Upewnij się, że foldery są poprawne.")
    exit()

def get_best_move(game, solver: Solver, depth: int) -> Move:
    state = game.state
    moves = state.get_moves()
    random.shuffle(moves) # Losowość przy remisach (wymagana w zadaniu)
    
    best_move = None
    best_value = -math.inf
    
    # Gracz, którego turę aktualnie symulujemy, chce zmaksymalizować swój wynik
    current_player = state.get_current_player()
    
    alpha = -math.inf
    beta = math.inf
    
    for move in moves:
        next_state = state.make_move(move)

        val = solver.alphabeta(next_state, depth - 1, alpha, beta, current_player)
        
        if val > best_value:
            best_value = val
            best_move = move
        
        # Optymalizacja alpha w korzeniu
        alpha = max(alpha, best_value)
        
    return best_move

def play_match(d1: int, d2: int, size: int = 3, verbose: bool = False) -> str:
    """
    Rozgrywa mecz: Gracz 1 (głębokość d1) vs Gracz 2 (głębokość d2).
    Zwraca '1', '2' lub 'Draw'.
    """
    game = DotsAndBoxes(size=size)
    solver = Solver()
    
    p1 = game.first_player
    p2 = game.second_player
    
    if verbose:
        print(f"\n--- Start Gry: P1(d={d1}) vs P2(d={d2}) ---")
        print(game)

    while not game.is_finished():
        curr = game.get_current_player()
        depth = d1 if curr.char == p1.char else d2
        
        move = get_best_move(game, solver, depth)
        game.make_move(move)
        
        if verbose:
            print(f"\nRuch gracza {curr.char}: {move.connection}{move.loc}")
            print(game)

    winner = game.get_winner()
    if verbose:
        print("\n--- Koniec ---")
        if winner:
            print(f"Wygrał: {winner.char}")
        else:
            print("Remis")
            
    return winner.char if winner else 'Draw'

def main():
    # KONFIGURACJA
    DEPTHS = [1,2,3,4,5,6,7,8,9,10] # Wartości N i M do przetestowania
    GAMES_PER_PAIR = 50
    BOARD_SIZE = 2
    
    results = []
    
    print("=== 1. TEST POJEDYNCZEJ GRY (printy) ===")
    play_match(1, 1, size=BOARD_SIZE, verbose=True)
    
    print("\n=== 2. TURNIEJ GLÓWNY ===")
    total_steps = len(DEPTHS) ** 2
    step = 0
    
    for d1 in DEPTHS:
        for d2 in DEPTHS:
            step += 1
            print(f"Symulacja {step}/{total_steps}: P1(d={d1}) vs P2(d={d2})...")
            
            p1_wins = 0
            p2_wins = 0
            draws = 0
            
            for _ in range(GAMES_PER_PAIR):
                res = play_match(d1, d2, size=BOARD_SIZE)
                if res == '1': p1_wins += 1
                elif res == '2': p2_wins += 1
                else: draws += 1
            
            results.append({
                'Głębokość P1': d1,
                'Głębokość P2': d2,
                'Wygrane P1': p1_wins,
                'Wygrane P2': p2_wins,
                'Remisy': draws,
                '% Wygranych P1': (p1_wins / GAMES_PER_PAIR) * 100
            })

    # Prezentacja wyników
    df = pd.DataFrame(results)
    
    print("\n=== WYNIKI TABELARYCZNE ===")
    print(df.to_string(index=False))
    
    # Heatmapa
    pivot_table = df.pivot(index="Głębokość P1", columns="Głębokość P2", values="% Wygranych P1")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100)
    plt.title("Procent zwycięstw Gracza 1 (Max)")
    plt.ylabel("Głębokość Gracza 1")
    plt.xlabel("Głębokość Gracza 2")
    
    plt.savefig("wyniki_heatmapa.png")
    print("\nHeatmapa zapisana jako 'wyniki_heatmapa.png'.")

if __name__ == "__main__":
    main()