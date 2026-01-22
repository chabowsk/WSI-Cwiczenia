import matplotlib.pyplot as plt
import numpy as np
from solver import train_agent

EPISODES = 400      
MAX_STEPS = 200    
WINDOW = 20     
EXPERIMENTAL_VALUES = [0.1, 0.3, 0.7, 0.9]

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def run_all_experiments():
    print("Rozpoczynanie eksperymentów w środowisku Fixed FourRooms...")

    # --- EKSPERYMENT 1: Szybkość uczenia (Beta) ---
    # Badamy wpływ parametru beta (learning rate)
    betas = EXPERIMENTAL_VALUES
    fixed_gamma = 0.9
    fixed_epsilon = 0.1
    
    plt.figure(figsize=(10, 6))
    for beta in betas:
        print(f"Trenowanie Beta={beta}...")
        rewards = train_agent(EPISODES, beta=beta, gamma=fixed_gamma, epsilon=fixed_epsilon, max_steps=MAX_STEPS)
        smoothed = moving_average(rewards, WINDOW)
        plt.plot(smoothed, label=f'Beta (lr) = {beta}')
        
    plt.title(f'Wpływ parametru Beta (Szybkość uczenia)\nGamma={fixed_gamma}, Epsilon={fixed_epsilon}')
    plt.xlabel('Epizody')
    plt.ylabel('Średnia nagroda (Rolling Avg)')
    plt.legend()
    plt.grid(True)
    plt.savefig('exp_beta.png')
    print("Zapisano exp_beta.png")

    # --- EKSPERYMENT 2: Współczynnik dyskontowania (Gamma) ---
    # Badamy wpływ gamma
    gammas = EXPERIMENTAL_VALUES
    fixed_beta = 0.1
    
    plt.figure(figsize=(10, 6))
    for gamma in gammas:
        print(f"Trenowanie Gamma={gamma}...")
        rewards = train_agent(EPISODES, beta=fixed_beta, gamma=gamma, epsilon=fixed_epsilon, max_steps=MAX_STEPS)
        smoothed = moving_average(rewards, WINDOW)
        plt.plot(smoothed, label=f'Gamma = {gamma}')
        
    plt.title(f'Wpływ parametru Gamma (Dyskontowanie)\nBeta={fixed_beta}, Epsilon={fixed_epsilon}')
    plt.xlabel('Epizody')
    plt.ylabel('Średnia nagroda')
    plt.legend()
    plt.grid(True)
    plt.savefig('exp_gamma.png')
    print("Zapisano exp_gamma.png")

    # --- EKSPERYMENT 3: Eksploracja (Epsilon) ---
    # Badamy wpływ epsilon
    epsilons = EXPERIMENTAL_VALUES
    fixed_gamma = 0.9
    
    plt.figure(figsize=(10, 6))
    for eps in epsilons:
        print(f"Trenowanie Epsilon={eps}...")
        rewards = train_agent(EPISODES, beta=fixed_beta, gamma=fixed_gamma, epsilon=eps, max_steps=MAX_STEPS)
        smoothed = moving_average(rewards, WINDOW)
        plt.plot(smoothed, label=f'Epsilon = {eps}')
        
    plt.title(f'Wpływ parametru Epsilon (Eksploracja)\nBeta={fixed_beta}, Gamma={fixed_gamma}')
    plt.xlabel('Epizody')
    plt.ylabel('Średnia nagroda')
    plt.legend()
    plt.grid(True)
    plt.savefig('exp_epsilon.png')
    print("Zapisano exp_epsilon.png")

    plt.show()

if __name__ == "__main__":
    run_all_experiments()