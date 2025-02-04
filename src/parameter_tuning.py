import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from differential_evolution import differential_evolution
from functions import CECFunctions, generate_rotation_matrix, generate_shift_vector

def parameter_tuning_experiment():
    F_values = [0.25, 0.5, 0.75]
    CR_values = [0.6, 0.7, 0.8]
    population_sizes = [100, 200, 300]
    max_generations = [100, 200, 300]
    stagnation_generations = [25, 30, 35]

    rounding = 1e-6

    # F_values = [0.25, 0.5]
    # CR_values = [0.6, 0.7]
    # population_sizes = [100]
    # max_generations = [100]
    # stagnation_generations = [25, 30]

    dim = 2
    repetitions = 10
    bounds = [(-5, 5)] * dim
    shift = generate_shift_vector(dim, bounds)
    rotation = generate_rotation_matrix(dim)
    theoretical_optimum = 0.0  

    def objective_func(x):
        return CECFunctions.shifted_rotated_griewank(x, shift, rotation)

    results = {}
    best_combination = None
    best_success_rate = -1

    for F, CR, pop_size, max_gen, stag_gen in product(F_values, CR_values, population_sizes, max_generations, stagnation_generations):
        key = (F, CR, pop_size, max_gen, stag_gen)
        results[key] = {
            "success_rates": [],
            "fitness_values": []
        }
        print(f"Testing F={F}, CR={CR}, Pop={pop_size}, Gen={max_gen}, Stag={stag_gen}")

        successes = 0
        for _ in range(repetitions):
            de = differential_evolution(
                objective_func,
                bounds,
                population_size=pop_size,
                F=F,
                CR=CR,
                max_generations=max_gen,
                stagnation_generations=stag_gen
            )
            _, fitness = de.optimize()
            
            success = np.abs(fitness - theoretical_optimum) < rounding
            successes += int(success)
            results[key]["fitness_values"].append(fitness)
        
        success_rate = (successes / repetitions) * 100
        results[key]["success_rates"].append(success_rate)
        
        avg_success_rate = np.mean(results[key]["success_rates"])
        if avg_success_rate > best_success_rate:
            best_success_rate = avg_success_rate
            best_combination = key

    return results, best_combination, best_success_rate

def plot_parameter_results(results):
    plt.figure(figsize=(15, 10))

    plt.subplot(231)
    F_values = sorted(set(k[0] for k in results.keys()))
    F_success_rates = {F: [] for F in F_values}
    for (F, _, _, _, _), values in results.items():
        F_success_rates[F].extend(values["success_rates"])

    plt.boxplot([F_success_rates[F] for F in F_values], labels=[f"F={F}" for F in F_values])
    plt.title("Impact of F Parameter")
    plt.ylabel("Success Rate (%)")

    plt.subplot(232)
    CR_values = sorted(set(k[1] for k in results.keys()))
    CR_success_rates = {CR: [] for CR in CR_values}
    for (_, CR, _, _, _), values in results.items():
        CR_success_rates[CR].extend(values["success_rates"])

    plt.boxplot([CR_success_rates[CR] for CR in CR_values], labels=[f"CR={CR}" for CR in CR_values])
    plt.title("Impact of CR Parameter")
    plt.ylabel("Success Rate (%)")

    plt.subplot(233)
    pop_sizes = sorted(set(k[2] for k in results.keys()))
    pop_success_rates = {pop: [] for pop in pop_sizes}
    for (_, _, pop, _, _), values in results.items():
        pop_success_rates[pop].extend(values["success_rates"])

    plt.boxplot([pop_success_rates[pop] for pop in pop_sizes], labels=[f"Pop={pop}" for pop in pop_sizes])
    plt.title("Impact of Population Size")
    plt.ylabel("Success Rate (%)")

    plt.subplot(234)
    gen_sizes = sorted(set(k[3] for k in results.keys()))
    gen_success_rates = {gen: [] for gen in gen_sizes}
    for (_, _, _, gen, _), values in results.items():
        gen_success_rates[gen].extend(values["success_rates"])

    plt.boxplot([gen_success_rates[gen] for gen in gen_sizes], labels=[f"Gen={gen}" for gen in gen_sizes])
    plt.title("Impact of Max Generations")
    plt.ylabel("Success Rate (%)")

    plt.subplot(235)
    stag_sizes = sorted(set(k[4] for k in results.keys()))
    stag_success_rates = {stag: [] for stag in stag_sizes}
    for (_, _, _, _, stag), values in results.items():
        stag_success_rates[stag].extend(values["success_rates"])

    plt.boxplot([stag_success_rates[stag] for stag in stag_sizes], labels=[f"Stag={stag}" for stag in stag_sizes])
    plt.title("Impact of Stagnation Generations")
    plt.ylabel("Success Rate (%)")

    plt.tight_layout()
    plt.savefig("parameter_tuning_results.png")
    plt.close()

if __name__ == "__main__":
    results, best_params, best_rate = parameter_tuning_experiment()
    plot_parameter_results(results)
    
    F, CR, pop_size, max_gen, stag_gen = best_params
    print("\nBest parameter combination:")
    print(f"F = {F}")
    print(f"CR = {CR}")
    print(f"Population Size = {pop_size}")
    print(f"Max Generations = {max_gen}")
    print(f"Stagnation Generations = {stag_gen}")
    print(f"Success Rate = {best_rate:.2f}%")