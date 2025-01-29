import matplotlib.pyplot as plt
from itertools import product
from differential_evolution import differential_evolution
from cec_functions import CECFunctions, generate_rotation_matrix, generate_shift_vector


def parameter_tuning_experiment():
    F_values = [0.4, 0.6, 0.8, 1.0]
    CR_values = [0.3, 0.5, 0.7, 0.9]
    population_sizes = [30, 50, 70, 100]

    dim = 10
    bounds = [(-100, 100)] * dim
    shift = generate_shift_vector(dim, bounds)
    rotation = generate_rotation_matrix(dim)

    def objective_func(x):
        return CECFunctions.shifted_rotated_griewank(x, shift, rotation)

    results = {}

    for F, CR, pop_size in product(F_values, CR_values, population_sizes):
        key = (F, CR, pop_size)
        results[key] = []
        print(f"Testing F={F}, CR={CR}, Pop={pop_size}")

        for _ in range(10):
            de = differential_evolution(
                objective_func,
                bounds,
                population_size=pop_size,
                F=F,
                CR=CR,
                max_generations=100,
            )
            _, fitness = de.optimize()
            results[key].append(fitness)

    return results


def plot_parameter_results(results):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    F_values = sorted(set(k[0] for k in results.keys()))
    F_results = {F: [] for F in F_values}
    for (F, _, _), values in results.items():
        F_results[F].extend(values)

    plt.boxplot([F_results[F] for F in F_values], labels=[f"F={F}" for F in F_values])
    plt.title("Impact of F Parameter")
    plt.ylabel("Fitness")

    plt.subplot(132)
    CR_values = sorted(set(k[1] for k in results.keys()))
    CR_results = {CR: [] for CR in CR_values}
    for (_, CR, _), values in results.items():
        CR_results[CR].extend(values)

    plt.boxplot(
        [CR_results[CR] for CR in CR_values], labels=[f"CR={CR}" for CR in CR_values]
    )
    plt.title("Impact of CR Parameter")

    plt.subplot(133)
    pop_sizes = sorted(set(k[2] for k in results.keys()))
    pop_results = {pop: [] for pop in pop_sizes}
    for (_, _, pop), values in results.items():
        pop_results[pop].extend(values)

    plt.boxplot(
        [pop_results[pop] for pop in pop_sizes],
        labels=[f"Pop={pop}" for pop in pop_sizes],
    )
    plt.title("Impact of Population Size")

    plt.tight_layout()
    plt.savefig("parameter_tuning_results.png")
    plt.close()


if __name__ == "__main__":
    results = parameter_tuning_experiment()
    plot_parameter_results(results)
