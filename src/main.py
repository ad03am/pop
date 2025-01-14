import numpy as np


class differential_evolution:
    def __init__(self, func, bounds, population_size=50, max_generations=100, F=0.8, CR=0.7):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR

    def evaluate(self, x: np.ndarray):
        return self.func(x)

    def initialize_population(self):
        population = np.random.rand(self.population_size, self.dim)
        for i in range(self.dim):
            population[:, i] = (self.bounds[i, 1] - self.bounds[i, 0]) * population[:, i] + self.bounds[i, 0]
        return population

    def optimize(self):
        population = self.initialize_population()
        fitness = np.array([self.evaluate(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for generation in range(self.max_generations):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant = a + self.F * (b - c)

                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.random() < self.CR or j == np.random.randint(self.dim):
                        trial[j] = mutant[j]

                for j in range(self.dim):
                    trial[j] = np.clip(trial[j], self.bounds[j, 0], self.bounds[j, 1])

                trial_fitness = self.evaluate(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness

        return best_solution, best_fitness
