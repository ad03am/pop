import matplotlib.pyplot as plt

def plot_comparison(standard_results, surrogate_results):
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.boxplot([standard_results, surrogate_results])
    plt.xticks([1,2], ['Standard', 'Surrogate'])
    plt.title('Wyniki')
    
    plt.savefig("comparison.png")

plot_comparison([1,3,4,4,7,6], [2,3,4,5,6])