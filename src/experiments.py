import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import differential_evolution
from surrogate_de import surrogate_de
from functions import CECFunctions, generate_rotation_matrix, generate_shift_vector


def run_experiments():
    dim = 10
    bounds = [(-100, 100)] * dim
    n_runs = 50

    shift = generate_shift_vector(dim, bounds)
    rotation = generate_rotation_matrix(dim)

    test_functions = {
        "Shifted Sphere": (
            lambda x: CECFunctions.shifted_sphere(x, shift),
            (-100, 100),
        ),
        "Shifted Schwefel": (
            lambda x: CECFunctions.ackley(x, shift),
            (-500, 500),
        ),
        "Shifted Rotated Elliptic": (
            lambda x: CECFunctions.shifted_rotated_high_conditioned_elliptic(
                x, shift, rotation
            ),
            (-100, 100),
        ),
        "Shifted Rotated Griewank": (
            lambda x: CECFunctions.shifted_rotated_griewank(x, shift, rotation),
            (-600, 600),
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

    for func_name, (func, bound) in test_functions.items():
        print(f"\nTesting on {func_name}...")
        bounds = [bound] * dim

        standard_results = []
        standard_convergence = []
        surrogate_results = []
        surrogate_convergence = []
        standard_evaluations = []
        surrogate_evaluations = []
        population_size = 30

        for i in range(n_runs):
            print(f"Run {i + 1}/{n_runs}")

            population = np.random.rand(population_size, dim)
            for i in range(dim):
                population[:][i] = (bounds[i][1] - bounds[i][0]) * population[:][i] + bounds[i][0]

            std_de = differential_evolution(func, bounds, population_size=population_size, population=population.copy())
            _, fitness = std_de.optimize()
            standard_results.append(fitness)
            standard_convergence.append(std_de.best_history)
            standard_evaluations.append(std_de.evaluations)

            sur_de = surrogate_de(func, bounds, population_size=population_size, population=population.copy())
            _, fitness = sur_de.optimize()
            surrogate_results.append(fitness)
            surrogate_convergence.append(sur_de.best_history)
            surrogate_evaluations.append(sur_de.evaluations)

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

        min_len = min(
            min(len(hist) for hist in standard_convergence),
            min(len(hist) for hist in surrogate_convergence),
        )

        std_conv = np.mean([hist[:min_len] for hist in standard_convergence], axis=0)
        sur_conv = np.mean([hist[:min_len] for hist in surrogate_convergence], axis=0)

        convergence_data[func_name] = {
            "standard": std_conv,
            "surrogate": sur_conv,
            "generations": np.arange(min_len),
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

    plot_results(results, convergence_data, all_evaluations_lists)


def plot_results(results, convergence_data, all_evaluations_lists):
    n_funcs = len(results)
    fig, axes = plt.subplots(2, n_funcs, figsize=(5 * n_funcs, 10))

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

    plt.tight_layout()
    plt.savefig("comprehensive_results.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_experiments()