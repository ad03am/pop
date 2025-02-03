import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import differential_evolution
from surrogate_de import surrogate_de
from functions import CECFunctions, generate_rotation_matrix, generate_shift_vector
from scipy import stats

def run_comparison(dim=2, n_runs=50):
    # Test configuration
    bounds = [(-100, 100)] * dim
    shift = generate_shift_vector(dim, bounds)
    rotation = generate_rotation_matrix(dim)
    top_percentages = [1.0, 0.5]
    population_size = 100
    
    # Dictionary to store results
    results = {
        "standard_de": {
            "fitness": [],
            "evaluations": [],
            "optima_found": []
        }
    }
    
    for top_perc in top_percentages:
        results[f"surrogate_de_{top_perc}"] = {
            "fitness": [],
            "evaluations": [],
            "optima_found": []
        }

    # Test functions with their theoretical optima
    test_functions = {
        # "Shifted Sphere": (
        #     lambda x: CECFunctions.shifted_sphere(x, shift),
        #     0.0
        # ),
        "Shifted Rotated Griewank": (
            lambda x: CECFunctions.shifted_rotated_griewank(x, shift, rotation),
            0.0
        )
    }

    for func_name, (func, theoretical_optimum) in test_functions.items():
        print(f"\nTesting on {func_name}...")
        
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")
            
            # Initialize population
            population = np.random.rand(population_size, dim)
            for i in range(dim):
                population[:, i] = (bounds[i][1] - bounds[i][0]) * population[:, i] + bounds[i][0]
            
            # Run standard DE
            std_de = differential_evolution(
                func, 
                bounds, 
                population_size=population_size,
                population=population.copy()
            )
            _, std_fitness = std_de.optimize()
            
            results["standard_de"]["fitness"].append(std_fitness)
            results["standard_de"]["evaluations"].append(std_de.evaluations)
            results["standard_de"]["optima_found"].append(
                np.abs(std_fitness - theoretical_optimum) < 1e-2
            )
            
            for top_perc in top_percentages:
                key = f"surrogate_de_{top_perc}"
                print(f"  Testing with top_percentage={top_perc}")
                sur_de = surrogate_de(
                    func, 
                    bounds, 
                    population_size=population_size,
                    population=population.copy(),
                    top_percentage=top_perc
                )
                _, sur_fitness = sur_de.optimize()
                
                results[key]["fitness"].append(sur_fitness)
                results[key]["evaluations"].append(sur_de.evaluations)
                results[key]["optima_found"].append(
                    np.abs(sur_fitness - theoretical_optimum) < 1e-2
                )

            # key = f"surrogate_de_{top_perc}"
            # results[key]["fitness"].append(1)
            # results[key]["evaluations"].append(1)
            # results[key]["optima_found"].append(
            #     np.abs(1 - theoretical_optimum) < 1e-6
            # )

        # Save results
        # save_results(results, func_name)
        
        # Create visualizations
        plot_results(results, func_name)
        
        # Perform statistical tests
        perform_statistical_analysis(results, func_name)

def save_results(results, func_name):
    with open(f'comparison_results_{func_name.lower().replace(" ", "_")}.txt', 'w') as f:
        f.write(f"Results for {func_name}\n")
        f.write("=" * 50 + "\n\n")
        
        for algorithm, data in results.items():
            f.write(f"\n{algorithm}:\n")
            f.write("-" * 30 + "\n")
            
            f.write(f"Fitness:\n")
            f.write(f"  Mean: {np.mean(data['fitness']):.2e}\n")
            f.write(f"  Std:  {np.std(data['fitness']):.2e}\n")
            f.write(f"  Best: {np.min(data['fitness']):.2e}\n")
            f.write(f"  Worst: {np.max(data['fitness']):.2e}\n\n")
            
            f.write(f"Evaluations:\n")
            f.write(f"  Mean: {np.mean(data['evaluations']):.2f}\n")
            f.write(f"  Std:  {np.std(data['evaluations']):.2f}\n")
            f.write(f"  Min:  {np.min(data['evaluations'])}\n")
            f.write(f"  Max:  {np.max(data['evaluations'])}\n\n")
            
            success_rate = np.mean(data['optima_found']) * 100
            f.write(f"Success Rate: {success_rate:.2f}%\n")
            
def plot_results(results, func_name):
    plt.figure(figsize=(15, 5))
    
    # Fitness comparison
    plt.subplot(131)
    data = [results[alg]["fitness"] for alg in results.keys()]
    plt.boxplot(data, labels=[alg.replace('_', ' ').title() for alg in results.keys()])
    plt.xticks(rotation=45)
    plt.ylabel('Fitness')
    plt.yscale('log')
    plt.title('Fitness Comparison')
    
    # Evaluations comparison
    plt.subplot(132)
    data = [results[alg]["evaluations"] for alg in results.keys()]
    plt.boxplot(data, labels=[alg.replace('_', ' ').title() for alg in results.keys()])
    plt.xticks(rotation=45)
    plt.ylabel('Number of Evaluations')
    plt.title('Evaluations Comparison')
    
    # Success rate comparison
    plt.subplot(133)
    success_rates = [np.mean(results[alg]["optima_found"]) * 100 for alg in results.keys()]
    plt.bar(range(len(results)), success_rates)
    plt.xticks(range(len(results)), [alg.replace('_', ' ').title() for alg in results.keys()], rotation=45)
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Comparison')
    
    plt.tight_layout()
    plt.savefig(f'comparison_results_{func_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def perform_statistical_analysis(results, func_name):
    with open(f'statistical_analysis_{func_name.lower().replace(" ", "_")}.txt', 'w') as f:
        f.write(f"Statistical Analysis for {func_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Reference for comparison (standard DE)
        std_fitness = results["standard_de"]["fitness"]
        std_evals = results["standard_de"]["evaluations"]
        
        for algorithm in results.keys():
            if algorithm == "standard_de":
                continue
                
            f.write(f"\nComparison: Standard DE vs {algorithm}\n")
            f.write("-" * 50 + "\n")
            
            # Fitness comparison
            stat, p_value = stats.mannwhitneyu(
                std_fitness,
                results[algorithm]["fitness"],
                alternative='two-sided'
            )
            f.write(f"\nFitness Mann-Whitney U test:\n")
            f.write(f"p-value: {p_value:.4e}\n")
            
            # Effect size (Cohen's d) for fitness
            d = (np.mean(std_fitness) - np.mean(results[algorithm]["fitness"])) / \
                np.sqrt((np.var(std_fitness) + np.var(results[algorithm]["fitness"])) / 2)
            f.write(f"Effect size (Cohen's d): {d:.4f}\n")
            
            # Evaluations comparison
            stat, p_value = stats.mannwhitneyu(
                std_evals,
                results[algorithm]["evaluations"],
                alternative='two-sided'
            )
            f.write(f"\nEvaluations Mann-Whitney U test:\n")
            f.write(f"p-value: {p_value:.4e}\n")
            
            # Effect size (Cohen's d) for evaluations
            d = (np.mean(std_evals) - np.mean(results[algorithm]["evaluations"])) / \
                np.sqrt((np.var(std_evals) + np.var(results[algorithm]["evaluations"])) / 2)
            f.write(f"Effect size (Cohen's d): {d:.4f}\n")
            
            # Success rate comparison
            stat, p_value = stats.fisher_exact([
                [sum(results["standard_de"]["optima_found"]), 
                 len(std_fitness) - sum(results["standard_de"]["optima_found"])],
                [sum(results[algorithm]["optima_found"]), 
                 len(results[algorithm]["optima_found"]) - sum(results[algorithm]["optima_found"])]
            ])
            f.write(f"\nSuccess rate Fisher's exact test:\n")
            f.write(f"p-value: {p_value:.4e}\n")

if __name__ == "__main__":
    run_comparison()