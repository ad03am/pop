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
        
        for gen in range(100):
            new_pop = np.zeros_like(pop)
            for i in range(len(pop)):
                a, b, c = pop[np.random.choice(len(pop), 3)]
                mutant = a + 0.8 * (b - c)
                new_pop[i] = mutant
            
            self.surrogate.train()
            pred = self.surrogate.predict(new_pop)
            
            if pred is not None:
                best_idx = np.argsort(pred)[:5]
                for idx in best_idx:
                    fitness[idx] = self.evaluate(new_pop[idx])
                    self.surrogate.add_sample(new_pop[idx], fitness[idx])
            
            pop = new_pop
            
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]
