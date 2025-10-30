import numpy as np
import matplotlib.pyplot as plt
from solver import Solver
from evaluation import calc_target

class GeneticSolver(Solver):
    
    def __init__(self, population_size=50, probability_of_mutation=0.00125, probability_of_crossover=0.7, generations=200):
        self.population_size = population_size
        self.probability_of_mutation = probability_of_mutation
        self.probability_of_crossover = probability_of_crossover
        self.generations = generations

    def get_parameters(self):
        return {
            "population_size": self.population_size,
            "mutation_rate": self.probability_of_mutation,
            "crossover_rate": self.probability_of_crossover,
            "generations": self.generations
        }

    def create_initial_population(self):
        population = np.random.randint(0, 2, (self.population_size, 1, 400))
        return population
    
    def roulette_wheel_selection(self, pop, fitness_scores, population_size):
        linear_shift = fitness_scores - np.min(fitness_scores) + 1e-6  # aby uniknac zerowych wartosci
        sum_linear_shift = np.sum(linear_shift)
        probabilities = linear_shift / sum_linear_shift
        selected = np.random.choice(population_size, size=population_size, p=probabilities.reshape(-1))
        for i in range(len(selected)):
            pop[i] = pop[selected[i]]
        return pop


    def mutation(self, pop, probability_of_mutation):
        individual_length = len(pop[0][0])
        for i in range(len(pop)):
            for j in range(individual_length):
                if np.random.uniform(0, 1) < probability_of_mutation:
                    pop[i][0][j] = 1 - pop[i][0][j]
        return pop
    

    def crossover(self, pop, probability_of_crossover):
            shuffled = np.random.permutation(pop) # wybranie losowej pary rodzicow
            for i in range(0, len(pop), 2):
                if np.random.uniform(0, 1) < probability_of_crossover:
                    point = np.random.randint(1, pop.shape[2]-1)
                    parent1 = shuffled[i]
                    parent2 = shuffled[i+1]
                    child1 = np.concatenate((parent1[:, :point], parent2[:, point:]), axis=1)
                    child2 = np.concatenate((parent2[:, :point], parent1[:, point:]), axis=1)
                    pop[i] = child1
                    pop[i+1] = child2
            return pop

    def calculate_fitness(self, evaluate, pop):
        fitness_scores = []
        for individual in range(self.population_size):
            fitness = evaluate(pop[individual])
            fitness_scores.append(fitness) 
        return fitness_scores
    
    def find_best_individual(self, pop, fitness_scores):
        best_idx = int(np.argmax(fitness_scores))
        best_individual = pop[best_idx]
        best_individual_score = fitness_scores[best_idx]
        return best_individual, best_individual_score
    
    def solve(self, evaluate, pop_init):
        pop = pop_init.copy()
    
        # 2. Ewaluacja populacji (calc_target)
        fitness_scores = self.calculate_fitness(evaluate, pop)

        # 3. Selekcja ruletkowa
        pop_sel = self.roulette_wheel_selection(pop, fitness_scores, self.population_size)
        # 4. Krzyżowanie jednopunktowe
        pop_m = self.crossover(pop_sel, self.probability_of_crossover)
        
        # 5. Mutacja
        pop_new = self.mutation(pop_m, self.probability_of_mutation)
        # 6. Nowa generacja (sukcesja)

        return pop_new

if __name__ == '__main__':
    solver = GeneticSolver()
    solutions = []
    values = [] 
    pop_size, mutation_rate, crossover_rate, generations = solver.get_parameters().values()

    pop_init = solver.create_initial_population()   
    fitness_scores = solver.calculate_fitness(calc_target, pop_init)
    best_individual, best_individual_score = solver.find_best_individual(pop_init, fitness_scores)
    solutions.append(best_individual)
    values.append(best_individual_score)
    pop = pop_init.copy()
    print(f"Generation 1: Best Score = {best_individual_score}")
    for generation in range(generations-1):
        pop = solver.solve(calc_target, pop)
        fitness_scores = solver.calculate_fitness(calc_target, pop)
        best_individual, best_individual_score = solver.find_best_individual(pop, fitness_scores)
        solutions.append(best_individual)
        values.append(best_individual_score)
        print(f"Generation {generation+2}: Best Score = {best_individual_score}")
    best_idx = np.argmax(values)
    best_score = values[best_idx]
    print("Best solution:", best_score, "z generacji", best_idx+1)
    print(solutions[best_idx])


    values_flat = [v.item() for v in values]

    mean_value = np.mean(values_flat)
    std_value = np.std(values_flat)
    best_value = np.max(values_flat)

    print("\n===== STATYSTYKI DLA PRZEBIEGU ALGORYTMU =====")
    print(f"🧬 Współczynnik mutacji: {mutation_rate}")
    print(f"📈 Średni wynik (mean): {mean_value:.6f}")
    print(f"⭐ Najlepszy wynik (max): {best_value:.6f}")
    print(f"📉 Odchylenie standardowe (std): {std_value:.6f}")
    print("=============================================\n")
    # Wykres
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(values_flat) + 1), values_flat, marker='o', markersize=3, linewidth=0, label="Najlepszy wynik")

    # Pomarańczowa linia odniesienia na poziomie -4
    plt.axhline(y=-4, color='orange', linestyle='--', linewidth=1, label='Wartość odniesienia (-4)')

    # Czerwony punkt o takim samym rozmiarze jak punkty na wykresie
    plt.scatter(best_idx + 1, best_score, color='red', s=9, label='Najlepszy osobnik', zorder=5)

    # Adnotacja tekstowa przy punkcie
    plt.text(best_idx + 1, best_score,
            f'  Gen {best_idx + 1}\n  {best_score}',
            color='red', fontsize=10, verticalalignment='bottom')

    plt.title("Najlepszy wynik w kolejnych generacjach", fontsize=14)
    plt.xlabel("Numer generacji", fontsize=12)
    plt.ylabel("Wartość funkcji celu (fitness)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()