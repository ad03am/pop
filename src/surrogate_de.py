from differential_evolution import differential_evolution
import numpy as np
from surrogate_model import surrogate_model


class surrogate_de(differential_evolution):
    def __init__(self, func, bounds, top_percentage=0.25, population_size=200, max_generations=200, F=0.5, CR=0.25, population=None):
        super().__init__(func, bounds, population_size, max_generations, F, CR)
        self.top_percentage = top_percentage
        self.population = population
        self.surrogate = surrogate_model()
        self.convergence_history = []
        self.best_history = []

    def optimize(self):
        if self.population is None:
            pop = self.initialize_population()
        else:
            pop = self.population
        fitness = np.array([self.evaluate(ind) for ind in pop])

        self.convergence_history = [np.mean(fitness)]
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx].copy()
        best_fitness = fitness[best_idx]
        self.best_history = [best_fitness]

        for ind, fit in zip(pop, fitness):
            self.surrogate.add_sample(ind, fit)

        for gen in range(self.max_generations):
            self.surrogate.train()
            new_pop = np.zeros_like(pop)
            new_fitness = np.full(self.population_size, np.inf)

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)

                trial = np.copy(pop[i])
                for j in range(self.dim):
                    if np.random.random() < self.CR or j == np.random.randint(self.dim):
                        trial[j] = mutant[j]
                    trial[j] = np.clip(trial[j], self.bounds[j][0], self.bounds[j][1])

                new_pop[i] = trial

            pred = self.surrogate.predict(new_pop)
            if pred is not None:
                best_candidates = np.argsort(pred)[: max(5, int(self.top_percentage * self.population_size))]

                for idx in best_candidates:
                    new_fitness[idx] = self.evaluate(new_pop[idx])
                    self.surrogate.add_sample(new_pop[idx], new_fitness[idx])

                    if new_fitness[idx] < best_fitness:
                        best_solution = new_pop[idx].copy()
                        best_fitness = new_fitness[idx]
            else:
                for i in range(self.population_size):
                    new_fitness[i] = self.evaluate(new_pop[i])
                    self.surrogate.add_sample(new_pop[i], new_fitness[i])

                    if new_fitness[i] < best_fitness:
                        best_solution = new_pop[i].copy()
                        best_fitness = new_fitness[i]

            for i in range(self.population_size):
                if new_fitness[i] < fitness[i]:
                    pop[i] = new_pop[i]
                    fitness[i] = new_fitness[i]

            self.convergence_history.append(np.mean(fitness))
            self.best_history.append(best_fitness)

        return best_solution, best_fitness
