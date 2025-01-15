import numpy as np
import matplotlib.pyplot as plt
from main import differential_evolution
from surrogate_de import surrogate_de
from functions import rastrigin

dim = 10
bounds = [(-5.12, 5.12)] * dim
n_runs = 30

print("Testing Standard DE...")
standard_results = []
standard_evals = []
for i in range(n_runs):
    de = differential_evolution(rastrigin, bounds)
    _, fitness = de.optimize()
    standard_results.append(fitness)
    standard_evals.append(de.evaluations)

print("Testing Surrogate DE...")
surrogate_results = []
surrogate_evals = []
for i in range(n_runs):
    de = surrogate_de(rastrigin, bounds)
    _, fitness = de.optimize()
    surrogate_results.append(fitness)
    surrogate_evals.append(de.evaluations)

plt.figure(figsize=(6, 5))
plt.boxplot([standard_results, surrogate_results], labels=['Standard DE', 'Surrogate DE'])
plt.title('Objective Function Value')
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig('results_fitness.png')
plt.close()

plt.figure(figsize=(6, 5))
means = [np.mean(standard_evals), np.mean(surrogate_evals)]
std_devs = [np.std(standard_evals), np.std(surrogate_evals)]

plt.bar(['Standard DE', 'Surrogate DE'], means, yerr=std_devs, capsize=5)
plt.title('Average Number of Evaluations')
plt.ylabel('Number of Evaluations')
plt.tight_layout()
plt.savefig('results_evaluations.png')
plt.close()

print("\nAverage results:")
print(f"Standard DE: {np.mean(standard_results):.2f}")
print(f"Surrogate DE: {np.mean(surrogate_results):.2f}")
print("\nAverage number of evaluations:")
print(f"Standard DE: {np.mean(standard_evals):.0f}")
print(f"Surrogate DE: {np.mean(surrogate_evals):.0f}")
