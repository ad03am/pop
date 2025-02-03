import numpy as np
from scipy import stats


class differential_evolution:
    def __init__(
        self,
        func,
        bounds,
        population_size=200,
        max_generations=200,
        F=0.5,
        CR=0.25,
        stagnation_generations=20,
        population=None,
    ):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR
        self.evaluations = 0
        self.stagnation_generations = stagnation_generations
        self.population = population
        self.convergence_history = []
        self.best_history = []

    def evaluate(self, x: np.ndarray):
        self.evaluations += 1
        return self.func(x)

    def initialize_population(self):
        population = np.random.rand(self.population_size, self.dim)
        for i in range(self.dim):
            population[:, i] = (self.bounds[i, 1] - self.bounds[i, 0]) * population[
                :, i
            ] + self.bounds[i, 0]
        return population

    def optimize(self):
        if self.population is None:
            population = self.initialize_population()
        else:
            population = self.population

        fitness = np.array([self.evaluate(ind) for ind in population])

        self.convergence_history = [np.mean(fitness)]
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        self.best_history = [best_fitness]
        no_improvement_count = 0
        last_best_fitness = best_fitness

        for generation in range(self.max_generations):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant = a + self.F * (b - c)
                trial = np.copy(population[i])

                for j in range(self.dim):
                    if np.random.random() < self.CR or j == np.random.randint(self.dim):
                        trial[j] = mutant[j]
                    trial[j] = np.clip(trial[j], self.bounds[j, 0], self.bounds[j, 1])

                trial_fitness = self.evaluate(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
                        no_improvement_count = 0

            if best_fitness >= last_best_fitness:
                no_improvement_count += 1
            else:
                last_best_fitness = best_fitness
                no_improvement_count = 0

            self.convergence_history.append(np.mean(fitness))
            self.best_history.append(best_fitness)

            if no_improvement_count >= self.stagnation_generations:
                print(
                    f"Stopped because of stagnation for {self.stagnation_generations} generations"
                )
                print(f"Generation: {generation + 1}/{self.max_generations}")
                break

        return best_solution, best_fitness
