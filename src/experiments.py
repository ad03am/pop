import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import differential_evolution
from surrogate_de import surrogate_de
from functions import CECFunctions, generate_rotation_matrix, generate_shift_vector

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
    all_algorithm_objects = {}
    
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

    for func_name, (func, bound, optimum) in test_functions.items():
        print(f"\nTesting on {func_name}...")
        bounds = [bound] * dim

        standard_results = []
        surrogate_results = []
        standard_evaluations = []
        surrogate_evaluations = []
        standard_hits = 0
        surrogate_hits = 0
        
        all_algorithm_objects[func_name] = {
            "standard": [],
            "surrogate": []
        }

        for i in range(n_runs):
            print(f"Run {i + 1}/{n_runs}")

            population = np.zeros((population_size, dim))
            for j in range(dim):
                space = np.linspace(bounds[j][0], bounds[j][1], 
                                int(np.ceil(np.power(population_size, 1/dim))))
                indices = np.random.choice(len(space), size=population_size)
                population[:, j] = space[indices]

            std_de = differential_evolution(func, bounds, population_size=population_size, 
                                         max_generations=max_generations, F=F, CR=CR, 
                                         stagnation_generations=stagnation_generations, 
                                         population=population.copy())
            _, fitness = std_de.optimize()
            standard_results.append(fitness)
            standard_evaluations.append(std_de.evaluations)
            all_algorithm_objects[func_name]["standard"].append(std_de)
            if np.abs(fitness - optimum) < rounding:
                standard_hits += 1

            sur_de = surrogate_de(func, bounds, top_percentage=top_percentage, 
                                population_size=population_size, max_generations=max_generations, 
                                F=F, CR=CR, stagnation_generations=stagnation_generations, 
                                population=population.copy(), surrogate_update_freq=surrogate_update_freq)
            _, fitness = sur_de.optimize()
            surrogate_results.append(fitness)
            surrogate_evaluations.append(sur_de.evaluations)
            all_algorithm_objects[func_name]["surrogate"].append(sur_de)
            if np.abs(fitness - optimum) < rounding:
                surrogate_hits += 1

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

    save_statistics(results, test_functions, all_evaluations, all_fitness, 
                   all_algorithm_objects, rounding, max_generations)
    
    plot_results(results, all_algorithm_objects, test_functions, 
                all_evaluations_lists, all_hits, rounding)
    
def save_statistics(results, test_functions, all_evaluations, all_fitness, 
                   all_algorithm_objects, rounding, max_generations):
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
            
            for alg_name, alg_key in [("Standard DE", "standard"), ("Surrogate DE", "surrogate")]:
                algorithms = all_algorithm_objects[func_name][alg_key]
                for gen in range(0, max_generations + 1, 100):
                    if gen == 0:
                        continue
                        
                    current_evals = []
                    success_count = 0
                    
                    for alg in algorithms:
                        for g, evals, fitness in alg.evaluation_history:
                            if g == gen:
                                current_evals.append(evals)
                                if np.abs(fitness - test_functions[func_name][2]) < rounding:
                                    success_count += 1
                                break
                    
                    if current_evals:
                        avg_evals = np.mean(current_evals)
                        success_rate = success_count / len(algorithms)
                        f.write(f"{alg_name} | {gen} | {avg_evals:.2f} | {success_rate:.2f}\n")


def plot_results(results, all_algorithm_objects, test_functions, all_evaluations_lists, all_hits, rounding):
    n_funcs = len(results)
    fig, axes = plt.subplots(3, n_funcs, figsize=(5 * n_funcs, 15))

    bar_width = 0.35
    algorithms = ["Standard DE", "Surrogate DE"]
    colors = ['blue', 'orange']

    for i, (func_name, func_results) in enumerate(results.items()):
        evaluations_to_optimum = {
            "Standard DE": [],
            "Surrogate DE": []
        }
        
        for run in range(len(all_algorithm_objects[func_name]["standard"])):
            std_fiteva = all_algorithm_objects[func_name]["standard"][run].fiteva
            sur_fiteva = all_algorithm_objects[func_name]["surrogate"][run].fiteva
            
            std_stop_eval = None
            sur_stop_eval = None
            
            for evals, fitness in sorted(std_fiteva.items()):
                if np.abs(fitness - test_functions[func_name][2]) < rounding:
                    std_stop_eval = evals
                    evaluations_to_optimum["Standard DE"].append(evals)
                    break
            else:
                evaluations_to_optimum["Standard DE"].append(max(std_fiteva.keys()))
                std_stop_eval = max(std_fiteva.keys())
            
            for evals, fitness in sorted(sur_fiteva.items()):
                if np.abs(fitness - test_functions[func_name][2]) < rounding:
                    sur_stop_eval = evals
                    evaluations_to_optimum["Surrogate DE"].append(evals)
                    break
            else:
                evaluations_to_optimum["Surrogate DE"].append(max(sur_fiteva.keys()))
                sur_stop_eval = max(sur_fiteva.keys())

            std_fiteva = sorted(std_fiteva.items())
            sur_fiteva = sorted(sur_fiteva.items())
            
            std_evals = [x[0] for x in std_fiteva if x[0] <= std_stop_eval]
            std_fits = [x[1] for x in std_fiteva if x[0] <= std_stop_eval]
            
            sur_evals = [x[0] for x in sur_fiteva if x[0] <= sur_stop_eval]
            sur_fits = [x[1] for x in sur_fiteva if x[0] <= sur_stop_eval]
            
            eval_points = np.linspace(0, max(max(std_evals), max(sur_evals)), 100)
            std_interpolated = np.interp(eval_points, std_evals, std_fits)
            sur_interpolated = np.interp(eval_points, sur_evals, sur_fits)
            
            std_mask = eval_points <= std_stop_eval
            sur_mask = eval_points <= sur_stop_eval
            
            std_final = np.full_like(eval_points, np.nan, dtype=float)
            sur_final = np.full_like(eval_points, np.nan, dtype=float)
            
            std_final[std_mask] = std_interpolated[std_mask]
            sur_final[sur_mask] = sur_interpolated[sur_mask]
            
            if run == 0:
                avg_fitness_std = std_final
                avg_fitness_sur = sur_final
                count_std = np.ones_like(eval_points)
                count_sur = np.ones_like(eval_points)
            else:
                mask_std = ~np.isnan(std_final)
                mask_sur = ~np.isnan(sur_final)
                
                avg_fitness_std[mask_std] += std_final[mask_std]
                avg_fitness_sur[mask_sur] += sur_final[mask_sur]
                count_std[mask_std] += 1
                count_sur[mask_sur] += 1
        
        means = [np.mean(evaluations_to_optimum["Standard DE"]), 
                np.mean(evaluations_to_optimum["Surrogate DE"])]
        stds = [np.std(evaluations_to_optimum["Standard DE"]), 
               np.std(evaluations_to_optimum["Surrogate DE"])]
        
        x = np.arange(len(algorithms))
        axes[0, i].bar(x, means, bar_width, yerr=stds, capsize=5, color=colors)
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels(algorithms)
        axes[0, i].set_title(f"{func_name}\nEvaluations to Reach Optimum")
        axes[0, i].set_ylabel("Number of Evaluations")

        avg_fitness_std = np.divide(avg_fitness_std, count_std, where=count_std>0, out=np.full_like(avg_fitness_std, np.nan))
        avg_fitness_sur = np.divide(avg_fitness_sur, count_sur, where=count_sur>0, out=np.full_like(avg_fitness_sur, np.nan))

        axes[1, i].plot(eval_points, avg_fitness_std, label="Standard DE", color=colors[0], alpha=0.8)
        axes[1, i].plot(eval_points, avg_fitness_sur, label="Surrogate DE", color=colors[1], alpha=0.8)
        axes[1, i].set_title("Convergence History")
        axes[1, i].set_xlabel("Number of Evaluations")
        axes[1, i].set_ylabel("Best Fitness")
        axes[1, i].set_yscale("log")
        axes[1, i].grid(True, which="both", ls="-", alpha=0.2)
        axes[1, i].legend()

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