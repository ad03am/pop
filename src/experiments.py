import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import differential_evolution
from surrogate_de import surrogate_de
from functions import CECFunctions, generate_rotation_matrix, generate_shift_vector


def run_experiments():
    dim = 10
    bounds = [(-100, 100)] * dim
    n_runs = 3

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

    for func_name, (func, bound) in test_functions.items():
        print(f"\nTesting on {func_name}...")
        bounds = [bound] * dim

        standard_results = []
        standard_convergence = []
        for i in range(n_runs):
            print(f"Standard DE - Run {i + 1}/{n_runs}")
            de = differential_evolution(func, bounds)
            _, fitness = de.optimize()
            standard_results.append(fitness)
            standard_convergence.append(de.convergence_history)

        surrogate_results = []
        surrogate_convergence = []
        for i in range(n_runs):
            print(f"Surrogate DE - Run {i + 1}/{n_runs}")
            de = surrogate_de(func, bounds)
            _, fitness = de.optimize()
            surrogate_results.append(fitness)
            surrogate_convergence.append(de.convergence_history)

        results[func_name] = {
            "standard": standard_results,
            "surrogate": surrogate_results,
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

        p_value = differential_evolution.statistical_test(
            standard_results, surrogate_results
        )
        print(f"\nStatistical test p-value: {p_value}")
        if p_value < 0.05:
            print("The difference is statistically significant")
        else:
            print("The difference is not statistically significant")

    plot_results(results, convergence_data)


def plot_results(results, convergence_data):
    n_funcs = len(results)
    fig, axes = plt.subplots(2, n_funcs, figsize=(5 * n_funcs, 10))

    for i, (func_name, func_results) in enumerate(results.items()):
        bp = axes[0, i].boxplot(
            [func_results["standard"], func_results["surrogate"]],
            labels=["Standard DE", "Surrogate DE"],
        )
        axes[0, i].set_title(f"{func_name}\nFinal Fitness Distribution")
        axes[0, i].set_yscale("log")

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
        axes[1, i].set_ylabel("Average Fitness")
        axes[1, i].set_yscale("log")
        axes[1, i].grid(True, which="both", ls="-", alpha=0.2)
        axes[1, i].legend()

    plt.tight_layout()
    plt.savefig("comprehensive_results.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_experiments()
