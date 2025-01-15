from main import differential_evolution
import numpy as np
from surrogate_model import surrogate_model


class surrogate_de(differential_evolution):
    def __init__(self, func, bounds):
        super().__init__(func, bounds)
        self.surrogate = surrogate_model()

    def optimize(self):
        pop = self.initialize_population()
        fitness = np.array([self.evaluate(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx].copy()
        best_fitness = fitness[best_idx]

        for ind, fit in zip(pop, fitness):
            self.surrogate.add_sample(ind, fit)

        for gen in range(100):
            self.surrogate.train()
            new_pop = np.zeros_like(pop)
            new_fitness = np.full(self.population_size, np.inf)
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + 0.8 * (b - c)
                for j in range(self.dim):
                    mutant[j] = np.clip(
                        mutant[j],
                        self.bounds[j][0],
                        self.bounds[j][1]
                    )
                new_pop[i] = mutant

            pred = self.surrogate.predict(new_pop)

            if pred is not None:
                best_idx = np.argsort(pred)[:5]
                for idx in best_idx:
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

        return best_solution, best_fitness
