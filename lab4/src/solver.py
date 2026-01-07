import numpy as np
import matplotlib.pyplot as plt

START_POINT_1D = [3.0]
START_POINT_2D = [4.0, -3.0]
SHIFT = 1e-15
MAX_ITERATIONS = 1000
INIT_LEARNING_RATE = 0.01


def ackley_function(x):
    x = np.array(x)
    n = len(x)

    sum_sq = np.sum(x**2)
    part_1 = -0.2 * np.sqrt(1/n * sum_sq)

    sum_cos = np.sum(np.cos(2 * np.pi * x))
    part_2 = 1/n * sum_cos

    ackley_value = -20 * np.exp(part_1) - np.exp(part_2) + 20 + np.e

    return ackley_value

def ackley_gradient(x):
    x = np.array(x)
    n = len(x)

    # --- CZĘŚĆ 1 ---
    # Dodanie SHIFT pozwala uniknąć dzielenia przez zero, gdy algorytm znajduje się w minimum (0,0)
    sqrt_term_inside = (1/n) * np.sum(x**2)
    sqrt_term = np.sqrt(sqrt_term_inside + SHIFT)

    exp_part1 = np.exp(-0.2 * sqrt_term)
    term1_grad = (4 * x * exp_part1) / (n * sqrt_term)

    # --- CZĘŚĆ 2 ---
    cos_term_inside = (1/n) * np.sum(np.cos(2 * np.pi * x)) 

    exp_part2 = np.exp(cos_term_inside)
    term2_grad = (2 * np.pi / n) * np.sin(2 * np.pi * x) * exp_part2

    return term1_grad + term2_grad

class GradientDescentOptimizer:
    def __init__(self, learning_rate=INIT_LEARNING_RATE, max_iterations=MAX_ITERATIONS):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def minimize(self, func=ackley_function, gradient_func=ackley_gradient, start_point=START_POINT_1D):
        current_point = np.array(start_point, dtype=float)
        value_history = []
        point_history = [] 
        
        for i in range(self.max_iterations):
            current_value = func(current_point)
            value_history.append(current_value)
            point_history.append(current_point.copy())
            
            grad = gradient_func(current_point)
            current_point = current_point - self.learning_rate * grad
            
            if np.any(np.abs(current_point) > 100): break

        final_value = func(current_point)
        value_history.append(final_value)
        point_history.append(current_point.copy())

        return current_point, value_history, point_history