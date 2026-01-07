import matplotlib.pyplot as plt
import numpy as np

from solver import ackley_function, GradientDescentOptimizer, START_POINT_1D, START_POINT_2D, MAX_ITERATIONS

if __name__ == "__main__":
    lrs_to_plot = [0.01, 0.05, 0.1, 0.5] 
    
    plt.style.use('seaborn-v0_8-whitegrid')

    # =================================================================
    # OKNO 1: PORÓWNANIE ZBIEŻNOŚCI (LOG SCALE)
    # =================================================================
    fig1, (ax1_conv, ax2_conv) = plt.subplots(1, 2, figsize=(14, 6))
    fig1.suptitle("OKNO 1: Zbieżność algorytmu (Skala Logarytmiczna)", fontsize=16)

    # Lewy: 1D
    for lr in lrs_to_plot:
        opt = GradientDescentOptimizer(learning_rate=lr, max_iterations=100)
        _, history, _ = opt.minimize(start_point=START_POINT_1D)
        ax1_conv.plot(history, label=f'LR={lr}', linewidth=1.5)
    
    ax1_conv.set_title("Zbieżność w 1D")
    ax1_conv.set_xlabel("Iteracja")
    ax1_conv.set_ylabel("f(x) [log]")
    ax1_conv.set_yscale('log')
    ax1_conv.legend()
    ax1_conv.grid(True, which="both", ls="-", alpha=0.4)

    # Prawy: 2D
    for lr in lrs_to_plot:
        opt = GradientDescentOptimizer(learning_rate=lr, max_iterations=100)
        _, history, _ = opt.minimize(start_point=START_POINT_2D)
        ax2_conv.plot(history, label=f'LR={lr}', linewidth=1.5)

    ax2_conv.set_title("Zbieżność w 2D")
    ax2_conv.set_xlabel("Iteracja")
    ax2_conv.set_yscale('log')
    ax2_conv.legend()
    ax2_conv.grid(True, which="both", ls="-", alpha=0.4)


    # =================================================================
    # OKNO 2: WIZUALIZACJA 1D + SZCZEGÓŁOWY RAPORT
    # =================================================================
    print("\n" + "="*115)
    print(f"{'RAPORT DLA OKNA 2 (Problemy 1D, punkt startowy [3.0])':^115}")
    print("="*115)
    print(f"{'LR':<6} | {'Iteracje':<9} | {'Punkt końcowy':<16} | {'Wartość końcowa':<18} | {'Najlepszy punkt':<16} | {'Najlepsza wartość':<18}")
    print("-" * 115)

    fig2, axs_1d = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle("OKNO 2: Trajektorie w 1D", fontsize=16)
    axs_1d_flat = axs_1d.flatten()

    x_range = np.linspace(-5, 5, 400)
    y_range = [ackley_function([val]) for val in x_range]

    for i, lr in enumerate(lrs_to_plot):
        ax = axs_1d_flat[i]
        
        # Tło
        ax.plot(x_range, y_range, color='gray', alpha=0.4)
        
        # Obliczenia
        opt_plot = GradientDescentOptimizer(learning_rate=lr, max_iterations=MAX_ITERATIONS) 
        _, _, points_plot = opt_plot.minimize(start_point=START_POINT_1D)
        
        # Rysowanie
        points_flat_plot = np.array(points_plot).flatten()
        vals_plot = [ackley_function([p]) for p in points_flat_plot]
        ax.plot(points_flat_plot, vals_plot, 'r.', alpha=0.8, linewidth=1, markersize=8)
        ax.set_title(f"LR = {lr}")
        ax.set_ylim(-1, 15)
        ax.set_xlim(-5, 5)
        
        # --- OBLICZENIA DO RAPORTU ---
        opt_full = GradientDescentOptimizer(learning_rate=lr, max_iterations=MAX_ITERATIONS)
        final_pt, history_vals, point_history = opt_full.minimize(start_point=START_POINT_1D)
        
        # 1. Dane końcowe
        final_pt_scalar = final_pt[0]
        final_val = ackley_function(final_pt)

        # 2. Dane najlepsze (Minimum w całej historii)
        min_val_idx = np.argmin(history_vals)
        best_val = history_vals[min_val_idx]
        best_pt = point_history[min_val_idx][0]
        
        print(f"{lr:<6} | {MAX_ITERATIONS:<9} | {final_pt_scalar:<16.4f} | {final_val:<18.6f} | {best_pt:<16.4f} | {best_val:<18.6f}")


    # =================================================================
    # OKNO 3: WIZUALIZACJA 2D + SZCZEGÓŁOWY RAPORT
    # =================================================================
    print("\n"*3 + "="*125)
    print(f"{'RAPORT DLA OKNA 3 (Problemy 2D, punkt startowy [4.0, -3.0])':^125}")
    print("="*125)

    print(f"{'LR':<6} | {'Iteracje':<9} | {'Punkt końcowy':<22} | {'Wartość końcowa':<18} | {'Najlepszy punkt':<22} | {'Najlepsza wartość':<18}")
    print("-" * 125)

    fig3, axs_2d = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle("OKNO 3: Trajektorie w 2D", fontsize=16)
    axs_2d_flat = axs_2d.flatten()

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = ackley_function([X[i, j], Y[i, j]])

    for i, lr in enumerate(lrs_to_plot):
        ax = axs_2d_flat[i]
        
        # Tło
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        
        opt_plot = GradientDescentOptimizer(learning_rate=lr, max_iterations=MAX_ITERATIONS)
        _, _, points_plot = opt_plot.minimize(start_point=START_POINT_2D)
        points_arr = np.array(points_plot)
        
        # Rysowanie
        ax.plot(points_arr[:, 0], points_arr[:, 1], 'r.', markersize=4, linewidth=1)
        ax.scatter(0, 0, c='yellow', marker='*', s=150, edgecolors='black', zorder=10) 
        ax.scatter(points_arr[0,0], points_arr[0,1], c='white', marker='o', s=50, edgecolors='black', zorder=10)
        ax.set_title(f"LR = {lr}")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        # --- OBLICZENIA DO RAPORTU ---
        opt_full = GradientDescentOptimizer(learning_rate=lr, max_iterations=MAX_ITERATIONS)
        final_pt, history_vals, point_history = opt_full.minimize(start_point=START_POINT_2D)

        # 1. Dane końcowe
        final_val = ackley_function(final_pt)
        final_pt_str = f"[{final_pt[0]:.2f}, {final_pt[1]:.2f}]"

        # 2. Dane najlepsze
        min_val_idx = np.argmin(history_vals)
        best_val = history_vals[min_val_idx]
        best_pt_raw = point_history[min_val_idx]
        best_pt_str = f"[{best_pt_raw[0]:.2f}, {best_pt_raw[1]:.2f}]"
        
        print(f"{lr:<6} | {1000:<9} | {final_pt_str:<22} | {final_val:<18.6f} | {best_pt_str:<22} | {best_val:<18.6f}")

    print("="*125 + "\n")
    plt.show()