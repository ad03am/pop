import matplotlib.pyplot as plt
import numpy as np
from differential_evolution import differential_evolution
from functions import CECFunctions, generate_rotation_matrix, generate_shift_vector


def parameter_tuning_experiment():
    dim = 2
    repetitions = 50
    bounds = [(-5, 5)] * dim
    shift = generate_shift_vector(dim, bounds)
    rotation = generate_rotation_matrix(dim)
    theoretical_optimum = 0.0

    def objective_func(x):
        return CECFunctions.shifted_rotated_griewank(x, shift, rotation)

    results = {}

    rounding = 1e-6

    base_F = 0.5
    base_CR = 0.8
    base_pop_size = 200
    base_max_gen = 100
    base_stagnation_generations = 40

    F_values = [0.25, 0.5, 0.75]
    for F in F_values:
        key = f"F_{F}"
        results[key] = []
        print(f"Testing F={F}")

        for _ in range(repetitions):
            de = differential_evolution(
                objective_func,
                bounds,
                population_size=base_pop_size,
                F=F,
                CR=base_CR,
                max_generations=base_max_gen,
                stagnation_generations=base_stagnation_generations,
            )
            _, fitness = de.optimize()
            success = np.abs(fitness - theoretical_optimum) < rounding
            results[key].append(int(success))

    CR_values = [0.6, 0.7, 0.8]
    for CR in CR_values:
        key = f"CR_{CR}"
        results[key] = []
        print(f"Testing CR={CR}")

        for _ in range(repetitions):
            de = differential_evolution(
                objective_func,
                bounds,
                population_size=base_pop_size,
                F=base_F,
                CR=CR,
                max_generations=base_max_gen,
                stagnation_generations=base_stagnation_generations,
            )
            _, fitness = de.optimize()
            success = np.abs(fitness - theoretical_optimum) < rounding
            results[key].append(int(success))

    population_sizes = [100, 200, 300]
    for pop_size in population_sizes:
        key = f"pop_{pop_size}"
        results[key] = []
        print(f"Testing population size={pop_size}")

        for _ in range(repetitions):
            de = differential_evolution(
                objective_func,
                bounds,
                population_size=pop_size,
                F=base_F,
                CR=base_CR,
                max_generations=base_max_gen,
                stagnation_generations=base_stagnation_generations,
            )
            _, fitness = de.optimize()
            success = np.abs(fitness - theoretical_optimum) < rounding
            results[key].append(int(success))

    max_generations = [100, 200, 300]
    for max_gen in max_generations:
        key = f"gen_{max_gen}"
        results[key] = []
        print(f"Testing max generations={max_gen}")

        for _ in range(repetitions):
            de = differential_evolution(
                objective_func,
                bounds,
                population_size=base_pop_size,
                F=base_F,
                CR=base_CR,
                max_generations=max_gen,
                stagnation_generations=base_stagnation_generations,
            )
            _, fitness = de.optimize()
            success = np.abs(fitness - theoretical_optimum) < rounding
            results[key].append(int(success))

    stagnation_generations = [25, 30, 35]
    for stagnation_gen in stagnation_generations:
        key = f"stagnation_{stagnation_gen}"
        results[key] = []
        print(f"Testing stagnation generations={stagnation_gen}")

        for _ in range(repetitions):
            de = differential_evolution(
                objective_func,
                bounds,
                population_size=base_pop_size,
                F=base_F,
                CR=base_CR,
                max_generations=base_max_gen,
                stagnation_generations=stagnation_gen,
            )
            _, fitness = de.optimize()
            success = np.abs(fitness - theoretical_optimum) < rounding
            results[key].append(int(success))

    return results


def plot_parameter_results(results):
    plt.figure(figsize=(15, 5))

    plt.subplot(151)
    F_results = [results[k] for k in results.keys() if k.startswith("F_")]
    F_labels = [k.split("_")[1] for k in results.keys() if k.startswith("F_")]
    success_rates = [np.mean(r) * 100 for r in F_results]
    plt.bar(range(len(success_rates)), success_rates)
    plt.xticks(range(len(success_rates)), [f"{f}" for f in F_labels])
    plt.title("Impact of F Parameter")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 100)

    plt.subplot(152)
    CR_results = [results[k] for k in results.keys() if k.startswith("CR_")]
    CR_labels = [k.split("_")[1] for k in results.keys() if k.startswith("CR_")]
    success_rates = [np.mean(r) * 100 for r in CR_results]
    plt.bar(range(len(success_rates)), success_rates)
    plt.xticks(range(len(success_rates)), [f"{cr}" for cr in CR_labels])
    plt.title("Impact of CR Parameter")
    plt.ylim(0, 100)

    plt.subplot(153)
    pop_results = [results[k] for k in results.keys() if k.startswith("pop_")]
    pop_labels = [k.split("_")[1] for k in results.keys() if k.startswith("pop_")]
    success_rates = [np.mean(r) * 100 for r in pop_results]
    plt.bar(range(len(success_rates)), success_rates)
    plt.xticks(range(len(success_rates)), [f"{p}" for p in pop_labels])
    plt.title("Impact of Population Size")
    plt.ylim(0, 100)

    plt.subplot(154)
    gen_results = [results[k] for k in results.keys() if k.startswith("gen_")]
    gen_labels = [k.split("_")[1] for k in results.keys() if k.startswith("gen_")]
    success_rates = [np.mean(r) * 100 for r in gen_results]
    plt.bar(range(len(success_rates)), success_rates)
    plt.xticks(range(len(success_rates)), [f"{g}" for g in gen_labels])
    plt.title("Impact of Max Generations")
    plt.ylim(0, 100)

    plt.subplot(155)
    stagnation_results = [results[k] for k in results.keys() if k.startswith("stagnation_")]
    stagnation_labels = [k.split("_")[1] for k in results.keys() if k.startswith("stagnation_")]
    success_rates = [np.mean(r) * 100 for r in stagnation_results]
    plt.bar(range(len(success_rates)), success_rates)
    plt.xticks(range(len(success_rates)), [f"{s}" for s in stagnation_labels])
    plt.title("Impact of Stagnation Generations")
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig("parameter_tuning_results_separate.png")
    plt.close()


if __name__ == "__main__":
    results = parameter_tuning_experiment()
    plot_parameter_results(results)