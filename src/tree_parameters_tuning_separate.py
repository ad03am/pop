import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import differential_evolution
from surrogate_de import surrogate_de
from custom_tree import DecisionTreeRegressor
from functions import CECFunctions, generate_rotation_matrix, generate_shift_vector

def tree_parameter_tuning_experiment():
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

    base_max_depth = 7
    base_min_samples_split = 5
    base_min_samples_leaf = 4

    base_top_percentage = 0.5
    base_surrogate_update_freq = 10

    max_depth_values = [5, 6, 7, 8]
    for max_depth in max_depth_values:
        key = f"depth_{max_depth}"
        results[key] = []
        print(f"Testing max_depth={max_depth}")

        for _ in range(repetitions):
            de = surrogate_de(objective_func, bounds, 
                            top_percentage=base_top_percentage,
                            surrogate_update_freq=base_surrogate_update_freq)
            de.surrogate.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=base_min_samples_split,
                min_samples_leaf=base_min_samples_leaf
            )
            _, fitness = de.optimize()
            success = np.abs(fitness - theoretical_optimum) < rounding
            results[key].append(int(success))

    min_samples_split_values = [4, 5, 6, 7]
    for min_split in min_samples_split_values:
        key = f"split_{min_split}"
        results[key] = []
        print(f"Testing min_samples_split={min_split}")

        for _ in range(repetitions):
            de = surrogate_de(objective_func, bounds, 
                            top_percentage=base_top_percentage,
                            surrogate_update_freq=base_surrogate_update_freq)
            de.surrogate.model = DecisionTreeRegressor(
                max_depth=base_max_depth,
                min_samples_split=min_split,
                min_samples_leaf=base_min_samples_leaf
            )
            _, fitness = de.optimize()
            success = np.abs(fitness - theoretical_optimum) < rounding
            results[key].append(int(success))

    min_samples_leaf_values = [4, 5, 6, 7]
    for min_leaf in min_samples_leaf_values:
        key = f"leaf_{min_leaf}"
        results[key] = []
        print(f"Testing min_samples_leaf={min_leaf}")

        for _ in range(repetitions):
            de = surrogate_de(objective_func, bounds, 
                            top_percentage=base_top_percentage,
                            surrogate_update_freq=base_surrogate_update_freq)
            de.surrogate.model = DecisionTreeRegressor(
                max_depth=base_max_depth,
                min_samples_split=base_min_samples_split,
                min_samples_leaf=min_leaf
            )
            _, fitness = de.optimize()
            success = np.abs(fitness - theoretical_optimum) < rounding
            results[key].append(int(success))

    return results

def plot_parameter_results(results):
    plt.figure(figsize=(20, 5))

    plt.subplot(141)
    depth_results = [results[k] for k in results.keys() if k.startswith("depth_")]
    depth_labels = [k.split("_")[1] for k in results.keys() if k.startswith("depth_")]
    success_rates = [np.mean(r) * 100 for r in depth_results]
    plt.bar(range(len(success_rates)), success_rates)
    plt.xticks(range(len(success_rates)), depth_labels)
    plt.title("Impact of max_depth")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 100)

    plt.subplot(142)
    split_results = [results[k] for k in results.keys() if k.startswith("split_")]
    split_labels = [k.split("_")[1] for k in results.keys() if k.startswith("split_")]
    success_rates = [np.mean(r) * 100 for r in split_results]
    plt.bar(range(len(success_rates)), success_rates)
    plt.xticks(range(len(success_rates)), split_labels)
    plt.title("Impact of min_samples_split")
    plt.ylim(0, 100)

    plt.subplot(143)
    leaf_results = [results[k] for k in results.keys() if k.startswith("leaf_")]
    leaf_labels = [k.split("_")[1] for k in results.keys() if k.startswith("leaf_")]
    success_rates = [np.mean(r) * 100 for r in leaf_results]
    plt.bar(range(len(success_rates)), success_rates)
    plt.xticks(range(len(success_rates)), leaf_labels)
    plt.title("Impact of min_samples_leaf")
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig("tree_parameter_tuning_separate_results.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_statistics(results):
    with open('tree_parameter_tuning_separate_statistics.txt', 'w') as f:
        for key in sorted(results.keys()):
            param_type, param_value = key.split('_')
            success_rate = np.mean(results[key]) * 100
            
            f.write(f"\nParameter: {param_type}, Value: {param_value}\n")
            f.write(f"Success Rate: {success_rate:.2f}%\n")

if __name__ == "__main__":
    results = tree_parameter_tuning_experiment()
    plot_parameter_results(results)
    save_statistics(results)