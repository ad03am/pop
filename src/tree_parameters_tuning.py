import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from differential_evolution import differential_evolution
from surrogate_de import surrogate_de
from custom_tree import DecisionTreeRegressor
from functions import CECFunctions, generate_rotation_matrix, generate_shift_vector

def run_tree_parameter_experiment():
    max_depth_values = [3, 5, 7, 9]
    min_samples_split_values = [4, 6, 8, 10]
    min_samples_leaf_values = [2, 5, 8, 11]
    
    dim = 2
    bounds = [(-5, 5)] * dim
    n_runs = 10
    
    shift = generate_shift_vector(dim, bounds)
    rotation = generate_rotation_matrix(dim)
    test_func = lambda x: CECFunctions.shifted_rotated_griewank(x, shift, rotation)
    
    results = {}

    for max_depth, min_split, min_leaf in product(max_depth_values, min_samples_split_values, min_samples_leaf_values):
        key = (max_depth, min_split, min_leaf)
        results[key] = []
        print(f"Testing max_depth={max_depth}, min_samples_split={min_split}, min_samples_leaf={min_leaf}")

        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")
            
            de = surrogate_de(test_func, bounds, top_percentage=0.3)
            de.surrogate.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf,
                max_generations=500
            )
            
            _, fitness = de.optimize()
            results[key].append({
                'fitness': fitness,
                'evaluations': de.evaluations
            })

    plot_parameter_results(results)
    save_statistics(results)
    return results

def plot_parameter_results(results):
    fig, axes = plt.subplots(2, 4, figsize=(20, 16))
    
    max_depth_values = sorted(set(k[0] for k in results.keys()))
    depth_fitness = {d: [] for d in max_depth_values}
    depth_evals = {d: [] for d in max_depth_values}
    for (depth, _, _, _), values in results.items():
        depth_fitness[depth].extend([v['fitness'] for v in values])
        depth_evals[depth].extend([v['evaluations'] for v in values])

    axes[0, 0].boxplot([depth_fitness[d] for d in max_depth_values], labels=max_depth_values)
    axes[0, 0].set_title('Impact of max_depth on Fitness')
    axes[0, 0].set_yscale('log')
    axes[1, 0].boxplot([depth_evals[d] for d in max_depth_values], labels=max_depth_values)
    axes[1, 0].set_title('Impact of max_depth on Evaluations')

    min_split_values = sorted(set(k[1] for k in results.keys()))
    split_fitness = {s: [] for s in min_split_values}
    split_evals = {s: [] for s in min_split_values}
    for (_, split, _, _), values in results.items():
        split_fitness[split].extend([v['fitness'] for v in values])
        split_evals[split].extend([v['evaluations'] for v in values])

    axes[0, 1].boxplot([split_fitness[s] for s in min_split_values], labels=min_split_values)
    axes[0, 1].set_title('Impact of min_samples_split on Fitness')
    axes[0, 1].set_yscale('log')
    axes[1, 1].boxplot([split_evals[s] for s in min_split_values], labels=min_split_values)
    axes[1, 1].set_title('Impact of min_samples_split on Evaluations')

    min_leaf_values = sorted(set(k[2] for k in results.keys()))
    leaf_fitness = {l: [] for l in min_leaf_values}
    leaf_evals = {l: [] for l in min_leaf_values}
    for (_, _, leaf, _), values in results.items():
        leaf_fitness[leaf].extend([v['fitness'] for v in values])
        leaf_evals[leaf].extend([v['evaluations'] for v in values])

    axes[0, 2].boxplot([leaf_fitness[l] for l in min_leaf_values], labels=min_leaf_values)
    axes[0, 2].set_title('Impact of min_samples_leaf on Fitness')
    axes[0, 2].set_yscale('log')
    axes[1, 2].boxplot([leaf_evals[l] for l in min_leaf_values], labels=min_leaf_values)
    axes[1, 2].set_title('Impact of min_samples_leaf on Evaluations')

    top_values = sorted(set(k[3] for k in results.keys()))
    top_fitness = {t: [] for t in top_values}
    top_evals = {t: [] for t in top_values}
    for (_, _, _, top), values in results.items():
        top_fitness[top].extend([v['fitness'] for v in values])
        top_evals[top].extend([v['evaluations'] for v in values])

    axes[0, 3].boxplot([top_fitness[t] for t in top_values], labels=top_values)
    axes[0, 3].set_title('Impact of top_percentage on Fitness')
    axes[0, 3].set_yscale('log')
    axes[1, 3].boxplot([top_evals[t] for t in top_values], labels=top_values)
    axes[1, 3].set_title('Impact of top_percentage on Evaluations')


    plt.tight_layout()
    plt.savefig('tree_parameter_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_statistics(results):
    with open('tree_parameter_statistics.txt', 'w') as f:
        for key in sorted(results.keys()):
            max_depth, min_split, min_leaf, top_perc = key
            fitness_values = [r['fitness'] for r in results[key]]
            eval_values = [r['evaluations'] for r in results[key]]
            
            f.write(f"\nmax_depth={max_depth}, min_samples_split={min_split}, min_samples_leaf={min_leaf}, top_percentage={top_perc}\n")
            f.write(f"Fitness - Mean: {np.mean(fitness_values):.2e}, Std: {np.std(fitness_values):.2e}\n")
            f.write(f"Evaluations - Mean: {np.mean(eval_values):.0f}, Std: {np.std(eval_values):.0f}\n")

if __name__ == "__main__":
    run_tree_parameter_experiment()