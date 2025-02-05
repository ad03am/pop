import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import differential_evolution
from surrogate_de import surrogate_de
from functions import CECFunctions, generate_rotation_matrix, generate_shift_vector
from itertools import product

F = 0.5
CR = 0.7
population_size = 200
max_generations = 300
top_percentage = 0.65
stagnation_generations = 30
surrogate_update_freq = 16

rounding = 1e-6

def run_experiments():
    dim = 2
    bounds = [(-5, 5)] * dim
    n_runs = 10

    shift = generate_shift_vector(dim, bounds)
    rotation = generate_rotation_matrix(dim)

    test_functions = {
        "Shifted Sphere": (
            lambda x: CECFunctions.shifted_sphere(x, shift),
            (-5, 5),
            0.0,
        ),
        "Shifted Schwefel": (
            lambda x: CECFunctions.ackley(x, shift),
            (-5, 5),
            0.0,
        ),
        "Shifted Rotated Elliptic": (
            lambda x: CECFunctions.shifted_rotated_high_conditioned_elliptic(
                x, shift, rotation
            ),
            (-5, 5),
            0.0,
        ),
        "Shifted Rotated Griewank": (
            lambda x: CECFunctions.shifted_rotated_griewank(x, shift, rotation),
            (-5, 5),
            0.0,
        ),
    }

    results = {}
    convergence_data = {}
    
    all_evaluations = {
        "Standard DE": {},
        "Surrogate DE": {}
    }
    all_fitness = {
        "Standard DE": {},
        "Surrogate DE": {}
    }
    all_evaluations_lists = {}
    all_hits = {}

    evaluation_history_data = {
        "Standard DE": {func_name: [] for func_name in test_functions.keys()},
        "Surrogate DE": {func_name: [] for func_name in test_functions.keys()}
    }

    for func_name, (func, bound, optimum) in test_functions.items():
        print(f"\nTesting on {func_name}...")
        bounds = [bound] * dim

        standard_results = []
        standard_convergence = []
        surrogate_results = []
        surrogate_convergence = []
        standard_evaluations = []
        surrogate_evaluations = []
        standard_hits = 0
        surrogate_hits = 0

        standard_evaluation_histories = []
        surrogate_evaluation_histories = []

        for i in range(n_runs):
            print(f"Run {i + 1}/{n_runs}")

            population = np.zeros((population_size, dim))
            
            for i in range(dim):
                space = np.linspace(bounds[i][0], bounds[i][1], 
                                int(np.ceil(np.power(population_size, 1/dim))))
                indices = np.random.choice(len(space), size=population_size)
                population[:, i] = space[indices]

            std_de = differential_evolution(func, bounds, population_size=population_size, max_generations=max_generations, F=F, CR=CR, stagnation_generations=stagnation_generations, population=population.copy())
            _, fitness = std_de.optimize()
            standard_results.append(fitness)
            standard_convergence.append(std_de.best_history)
            standard_evaluations.append(std_de.evaluations)
            standard_evaluation_histories.append(std_de.evaluation_history)
            if np.abs(fitness - optimum) < rounding:
                standard_hits += 1

            sur_de = surrogate_de(func, bounds, top_percentage=top_percentage, population_size=population_size, max_generations=max_generations, F=F, CR=CR, stagnation_generations=stagnation_generations, population=population.copy(), surrogate_update_freq=surrogate_update_freq)
            _, fitness = sur_de.optimize()
            surrogate_results.append(fitness)
            surrogate_convergence.append(sur_de.best_history)
            surrogate_evaluations.append(sur_de.evaluations)
            surrogate_evaluation_histories.append(sur_de.evaluation_history)
            if np.abs(fitness - optimum) < rounding:
                surrogate_hits += 1

        evaluation_history_data["Standard DE"][func_name] = standard_evaluation_histories
        evaluation_history_data["Surrogate DE"][func_name] = surrogate_evaluation_histories

        results[func_name] = {
            "standard": standard_results,
            "surrogate": surrogate_results,
        }

        all_evaluations["Standard DE"][func_name] = np.mean(standard_evaluations)
        all_evaluations["Surrogate DE"][func_name] = np.mean(surrogate_evaluations)
        all_fitness["Standard DE"][func_name] = np.mean(standard_results)
        all_fitness["Surrogate DE"][func_name] = np.mean(surrogate_results)
        all_evaluations_lists[func_name] = {
            "standard": standard_evaluations,
            "surrogate": surrogate_evaluations
        }
        all_hits[func_name] = {
            "standard": standard_hits / n_runs,
            "surrogate": surrogate_hits / n_runs
        }

        std_conv = np.full(max_generations, np.nan)
        sur_conv = np.full(max_generations, np.nan)

        for gen in range(max_generations):
            valid_std = [hist[gen] for hist in standard_convergence if gen < len(hist)]
            valid_sur = [hist[gen] for hist in surrogate_convergence if gen < len(hist)]
            
            if valid_std:
                std_conv[gen] = np.mean(valid_std)
            if valid_sur:
                sur_conv[gen] = np.mean(valid_sur)

        convergence_data[func_name] = {
            "standard": std_conv,
            "surrogate": sur_conv,
            "generations": np.arange(max_generations),
        }

    with open('algorithm_statistics.txt', 'w') as f:
        f.write("Average Number of Evaluations:\n")
        f.write("Algorithm | Shifted Sphere | Shifted Schwefel | Shifted Rotated Elliptic | Shifted Rotated Griewank\n")
        f.write("-" * 100 + "\n")
        
        for alg in ["Standard DE", "Surrogate DE"]:
            f.write(f"{alg} | ")
            f.write(" | ".join(f"{all_evaluations[alg][func_name]:.2f}" 
                             for func_name in test_functions.keys()))
            f.write("\n")
        
        f.write("\n\n")
        
        f.write("Average Fitness Values:\n")
        f.write("Algorithm | Shifted Sphere | Shifted Schwefel | Shifted Rotated Elliptic | Shifted Rotated Griewank\n")
        f.write("-" * 100 + "\n")
        
        for alg in ["Standard DE", "Surrogate DE"]:
            f.write(f"{alg} | ")
            f.write(" | ".join(f"{all_fitness[alg][func_name]:.6f}" 
                             for func_name in test_functions.keys()))
            f.write("\n")
        
        f.write("\n\n")
        f.write("Statistics every 100 generations:\n\n")
        
        for func_name in test_functions.keys():
            f.write(f"\n{func_name}:\n")
            f.write("Algorithm | Generation | Avg Evaluations | Success Rate\n")
            f.write("-" * 60 + "\n")
            
            for alg in ["Standard DE", "Surrogate DE"]:
                histories = evaluation_history_data[alg][func_name]
                unique_generations = sorted(set(gen for hist in histories for gen, _, _ in hist))
                
                for gen in unique_generations:
                    if gen % 100 == 0 or gen == max(unique_generations):
                        gen_data = []
                        for hist in histories:
                            matching_entries = [entry for entry in hist if entry[0] == gen]
                            if matching_entries:
                                gen_data.append(matching_entries[0])
                        
                        if gen_data:
                            avg_evals = np.mean([entry[1] for entry in gen_data])
                            success_rate = np.mean([np.abs(entry[2] - test_functions[func_name][2]) < rounding 
                                                  for entry in gen_data])
                            f.write(f"{alg} | {gen} | {avg_evals:.2f} | {success_rate:.2f}\n")

    plot_results(results, convergence_data, all_evaluations_lists, all_hits)

def plot_results(results, convergence_data, all_evaluations_lists, all_hits):
    n_funcs = len(results)
    fig, axes = plt.subplots(3, n_funcs, figsize=(5 * n_funcs, 15))

    bar_width = 0.35
    algorithms = ["Standard DE", "Surrogate DE"]
    colors = ['blue', 'orange']

    for i, (func_name, func_results) in enumerate(results.items()):
        std_mean = np.mean(all_evaluations_lists[func_name]["standard"])
        std_std = np.std(all_evaluations_lists[func_name]["standard"])
        sur_mean = np.mean(all_evaluations_lists[func_name]["surrogate"])
        sur_std = np.std(all_evaluations_lists[func_name]["surrogate"])

        means = [std_mean, sur_mean]
        stds = [std_std, sur_std]
        
        x = np.arange(len(algorithms))
        axes[0, i].bar(x, means, bar_width, yerr=stds, capsize=5, color=colors)
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels(algorithms)
        axes[0, i].set_title(f"{func_name}\nNumber of Evaluations")

        conv_data = convergence_data[func_name]
        axes[1, i].plot(
            conv_data["generations"],
            conv_data["standard"],
            label="Standard DE",
            alpha=0.8,
        )
        axes[1, i].plot(
            conv_data["generations"],
            conv_data["surrogate"],
            label="Surrogate DE",
            alpha=0.8,
        )
        axes[1, i].set_title("Convergence History")
        axes[1, i].set_xlabel("Generation")
        axes[1, i].set_ylabel("Best Fitness")
        axes[1, i].set_yscale("log")
        axes[1, i].grid(True, which="both", ls="-", alpha=0.2)
        axes[1, i].legend()
        axes[1, i].set_xlim(0, max_generations)

        axes[2, i].bar(algorithms, 
                      [all_hits[func_name]["standard"], 
                       all_hits[func_name]["surrogate"]], 
                      color=colors)
        axes[2, i].set_title("Success Rate")
        axes[2, i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("comprehensive_results.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    run_experiments()