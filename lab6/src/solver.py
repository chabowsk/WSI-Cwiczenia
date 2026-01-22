import gymnasium as gym
import minigrid
import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, action_space_n=3, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.action_space_n = action_space_n
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_space_n))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space_n))
        
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        actions_with_max_q = np.where(q_values == max_q)[0]
        return np.random.choice(actions_with_max_q)

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            next_max_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * next_max_q
        
        self.q_table[state][action] += self.lr * (target - current_q)

def get_simple_state(env):
    # Prosty stan: pozycja agenta (x, y) i kierunek
    return (*env.unwrapped.agent_pos, env.unwrapped.agent_dir)

def train_agent(episodes, beta, gamma, epsilon, max_steps=100):
    # Tworzenie środowiska zgodnie z instrukcją
    env = gym.make("MiniGrid-FourRooms-v0", max_steps=max_steps, agent_pos=(7, 7), goal_pos=(2, 2))
    
    agent = QLearningAgent(3, beta, gamma, epsilon)
    rewards_history = []
    
    for ep in range(episodes):
        env.reset()
        state = get_simple_state(env)
        total_reward = 0
        
        for _ in range(max_steps):
            action = agent.choose_action(state)
            
            # Standardowa nagroda Minigrid (bez modyfikacji)
            _, reward, terminated, truncated, _ = env.step(action)
            
            next_state = get_simple_state(env)

            # Modyfikacja nagrody: kara za każdy krok poza celem
            current_reward = reward if reward >0 else -0.01
            
            agent.learn(state, action, current_reward, next_state, terminated)
            
            state = next_state
            total_reward += reward
            
            if (terminated or truncated):
                break
        
        rewards_history.append(total_reward)
        
    env.close()
    return rewards_history