import numpy as np
from solver import Solver
from evaluation import calc_target

class GeneticSolver(Solver):
    
    def __init__(self, population_size=100, probability_of_mutation=0.02, probability_of_crossover=0.5, generations=100):
        self.population_size = population_size
        self.probability_of_mutation = probability_of_mutation
        self.probability_of_crossover = probability_of_crossover
        self.generations = generations

    def create_initial_population(self):
        population = np.random.randint(1, 2, (self.population_size, 1, 400)) # dodaj moze cos zamiast 400 
        return population

    def get_parameters(self):
        return {
            "population_size": self.population_size,
            "mutation_rate": self.probability_of_mutation,
            "crossover_rate": self.probability_of_crossover,
            "generations": self.generations
        }
    
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

    def solve(self, evaluate, pop_init):
        pop = pop_init.copy()
        for generation in range(self.generations):
            # print("generation:", generation)
            fitness_scores = []
            # 2. Ewaluacja populacji (calc_target)
            for individual in range(self.population_size):
                fitness = evaluate(pop[individual])
                fitness_scores.append(fitness)


            # 3. Selekcja ruletkowa
            pop_sel = self.roulette_wheel_selection(pop, fitness_scores, self.population_size)
            # 4. Krzyżowanie jednopunktowe
            pop_m = self.crossover(pop_sel, self.probability_of_crossover)
            
            # 5. Mutacja
            pop_new = self.mutation(pop_m, self.probability_of_mutation)
            # 6. Nowa generacja (sukcesja)

            pop = pop_new.copy()

        best_idx = int(np.argmax(fitness_scores))     #zwrócenie indeksu najlepszego 
        print("Best individual index:", best_idx, "with fitness:", fitness_scores[best_idx])
        return fitness_scores

if __name__ == '__main__':
    solver = GeneticSolver()
    solutions = []
    values = [] 
    pop_size, mutation_rate, crossover_rate, generations = solver.get_parameters().values()

    pop_init = solver.create_initial_population()   
    result = solver.solve(calc_target, pop_init)


   
   
   
   # TEST
    # pop_test = np.random.randint(0, 2, (10, 1, 5))
    # print("Test population:", pop_test)
    # fitness_test = [0.92, 0.07, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    # mutated_test = solver.mutation(pop_test,mutation_rate)
    # print("Selected zmutowane:", mutated_test)
