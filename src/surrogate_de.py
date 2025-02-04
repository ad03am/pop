from differential_evolution import differential_evolution
import numpy as np
from surrogate_model import surrogate_model


class surrogate_de(differential_evolution):
    def __init__(self, func, bounds, top_percentage=0.25, population_size=200, max_generations=100, 
                 F=0.5, CR=0.8, stagnation_generations=30, population=None, surrogate_update_freq=5):
        super().__init__(func, bounds, population_size, max_generations, F, CR, stagnation_generations, population)
        self.top_percentage = top_percentage
        self.population = population
        self.surrogate = surrogate_model()
        self.convergence_history = []
        self.best_history = []
        self.evaluation_history = []
        self.surrogate_update_freq = surrogate_update_freq
        self.stagnation_generations = stagnation_generations

    def optimize(self):
        if self.population is None:
            pop = self.initialize_population()
        else:
            pop = self.population
            
        fitness = np.array([self.evaluate(ind) for ind in pop])
        for ind, fit in zip(pop, fitness):
            self.surrogate.add_sample(ind, fit)

        self.convergence_history = [np.mean(fitness)]
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx].copy()
        best_fitness = fitness[best_idx]
        self.best_history = [best_fitness]
        no_improvement_count = 0
        last_best_fitness = best_fitness

        self.evaluation_history.append((0, self.evaluations, best_fitness))

        self.surrogate.train()

        global_best_solution = best_solution.copy()
        global_best_fitness = best_fitness
        
        for gen in range(self.max_generations):
            if gen % self.surrogate_update_freq == 0:
                self.surrogate.train()
                
            idxs = np.random.randint(0, self.population_size, size=(self.population_size, 3))
            for i in range(self.population_size):
                idxs[i] = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 
                    3, 
                    replace=False
                )
            
            a, b, c = pop[idxs[:, 0]], pop[idxs[:, 1]], pop[idxs[:, 2]]
            mutants = a + self.F * (b - c)
            
            mask = np.random.rand(self.population_size, self.dim) < self.CR
            trial = np.where(mask, mutants, pop)
            
            trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])
            
            pred_fitness = self.surrogate.predict(trial)
            
            if pred_fitness is not None:
                n_best = max(5, int(self.top_percentage * self.population_size))
                best_candidates = np.argsort(pred_fitness)[:n_best]
                
                for idx in best_candidates:
                    trial_fitness = self.evaluate(trial[idx])
                    self.surrogate.add_sample(trial[idx], trial_fitness)
                    
                    if trial_fitness < fitness[idx]:
                        pop[idx] = trial[idx]
                        fitness[idx] = trial_fitness
                        
                        if trial_fitness < best_fitness:
                            best_solution = trial[idx].copy()
                            best_fitness = trial_fitness
            else:
                for i in range(self.population_size):
                    trial_fitness = self.evaluate(trial[i])
                    self.surrogate.add_sample(trial[i], trial_fitness)
                    
                    if trial_fitness < fitness[i]:
                        pop[i] = trial[i]
                        fitness[i] = trial_fitness
                        
                        if trial_fitness < best_fitness:
                            best_solution = trial[i].copy()
                            best_fitness = trial_fitness
            
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < global_best_fitness:
                global_best_fitness = fitness[current_best_idx]
                global_best_solution = pop[current_best_idx].copy()
            
            if best_fitness >= last_best_fitness:
                no_improvement_count += 1
            else:
                last_best_fitness = best_fitness
                no_improvement_count = 0

            self.convergence_history.append(np.mean(fitness))
            self.best_history.append(global_best_fitness)

            if (gen + 1) % 100 == 0:
                self.evaluation_history.append((gen + 1, self.evaluations, best_fitness))

            if no_improvement_count >= self.stagnation_generations:
                print(f"Stopped because of stagnation for {self.stagnation_generations} generations")
                print(f"Generation: {gen + 1}/{self.max_generations}")
                if (gen + 1) % 100 != 0:
                    self.evaluation_history.append((gen + 1, self.evaluations, best_fitness))
                break

        if self.max_generations % 100 != 0:
            self.evaluation_history.append((self.max_generations, self.evaluations, best_fitness))

        return best_solution, best_fitness
