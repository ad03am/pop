import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import differential_evolution
from surrogate_de import surrogate_de
from custom_tree import DecisionTreeRegressor
from functions import CECFunctions, generate_rotation_matrix, generate_shift_vector

def surrogate_de_parameter_tuning_experiment():
    dim = 2
    repetitions = 50
    bounds = [(-5, 5)] * dim
    shift = generate_shift_vector(dim, bounds)
    rotation = generate_rotation_matrix(dim)
    theoretical_optimum = 0.0
    rounding = 1e-6

    def objective_func(x):
        return CECFunctions.shifted_rotated_griewank(x, shift, rotation)

    results = {}
    evaluations = {}

    base_max_depth = 7
    base_min_samples_split = 6
    base_min_samples_leaf = 5

    top_percentage_values = [0.55, 0.6, 0.65, 0.7]
    base_surrogate_update_freq = 15

    for top_pct in top_percentage_values:
        key = f"top_pct_{top_pct}"
        results[key] = []
        evaluations[key] = []
        print(f"Testing top_percentage={top_pct}")

        for _ in range(repetitions):
            de = surrogate_de(objective_func, bounds, 
                            top_percentage=top_pct,
                            surrogate_update_freq=base_surrogate_update_freq)
            de.surrogate.model = DecisionTreeRegressor(
                max_depth=base_max_depth,
                min_samples_split=base_min_samples_split,
                min_samples_leaf=base_min_samples_leaf
            )
            _, fitness = de.optimize()
            success = np.abs(fitness - theoretical_optimum) < rounding
            results[key].append(int(success))
            evaluations[key].append(de.evaluations)

    surrogate_update_freq_values = [15, 16, 17, 18]
    base_top_percentage = 0.7

    for update_freq in surrogate_update_freq_values:
        key = f"freq_{update_freq}"
        results[key] = []
        evaluations[key] = []
        print(f"Testing surrogate_update_freq={update_freq}")

        for _ in range(repetitions):
            de = surrogate_de(objective_func, bounds, 
                            top_percentage=base_top_percentage,
                            surrogate_update_freq=update_freq)
            de.surrogate.model = DecisionTreeRegressor(
                max_depth=base_max_depth,
                min_samples_split=base_min_samples_split,
                min_samples_leaf=base_min_samples_leaf
            )
            _, fitness = de.optimize()
            success = np.abs(fitness - theoretical_optimum) < rounding
            results[key].append(int(success))
            evaluations[key].append(de.evaluations)

    return results, evaluations

def plot_parameter_results(results, evaluations):
    plt.figure(figsize=(12, 10))

    plt.subplot(221)
    top_pct_results = [results[k] for k in results.keys() if k.startswith("top_pct_")]
    top_pct_labels = [k.split("_")[2] for k in results.keys() if k.startswith("top_pct_")]
    success_rates = [np.mean(r) * 100 for r in top_pct_results]
    plt.bar(range(len(success_rates)), success_rates)
    plt.xticks(range(len(success_rates)), top_pct_labels)
    plt.title("Success Rate vs top_percentage")
    plt.xlabel("top_percentage value")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 100)

    plt.subplot(222)
    freq_results = [results[k] for k in results.keys() if k.startswith("freq_")]
    freq_labels = [k.split("_")[1] for k in results.keys() if k.startswith("freq_")]
    success_rates = [np.mean(r) * 100 for r in freq_results]
    plt.bar(range(len(success_rates)), success_rates)
    plt.xticks(range(len(success_rates)), freq_labels)
    plt.title("Success Rate vs surrogate_update_freq")
    plt.xlabel("surrogate_update_freq value")
    plt.ylim(0, 100)

    plt.subplot(223)
    top_pct_evals = [evaluations[k] for k in evaluations.keys() if k.startswith("top_pct_")]
    mean_evals = [np.mean(e) for e in top_pct_evals]
    std_evals = [np.std(e) for e in top_pct_evals]
    plt.bar(range(len(mean_evals)), mean_evals, yerr=std_evals)
    plt.xticks(range(len(mean_evals)), top_pct_labels)
    plt.title("Function Evaluations vs top_percentage")
    plt.xlabel("top_percentage value")
    plt.ylabel("Number of Evaluations")

    plt.subplot(224)
    freq_evals = [evaluations[k] for k in evaluations.keys() if k.startswith("freq_")]
    mean_evals = [np.mean(e) for e in freq_evals]
    std_evals = [np.std(e) for e in freq_evals]
    plt.bar(range(len(mean_evals)), mean_evals, yerr=std_evals)
    plt.xticks(range(len(mean_evals)), freq_labels)
    plt.title("Function Evaluations vs surrogate_update_freq")
    plt.xlabel("surrogate_update_freq value")

    plt.tight_layout()
    plt.savefig("surrogate_de_parameter_tuning_results.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_statistics(results, evaluations):
    with open('surrogate_de_parameter_tuning_statistics.txt', 'w') as f:
        for key in sorted(results.keys()):
            if key.startswith("top_pct_"):
                param_type = "top_percentage"
                param_value = key.split('_')[2]
            else:
                param_type = "surrogate_update_freq"
                param_value = key.split('_')[1]
                
            success_rate = np.mean(results[key]) * 100
            mean_evals = np.mean(evaluations[key])
            std_evals = np.std(evaluations[key])
            
            f.write(f"\nParameter: {param_type}, Value: {param_value}\n")
            f.write(f"Success Rate: {success_rate:.2f}%\n")
            f.write(f"Mean Evaluations: {mean_evals:.2f} ± {std_evals:.2f}\n")

if __name__ == "__main__":
    results, evaluations = surrogate_de_parameter_tuning_experiment()
    plot_parameter_results(results, evaluations)
    save_statistics(results, evaluations)