import numpy as np
from scipy.optimize import minimize
from functions import CECFunctions, generate_rotation_matrix, generate_shift_vector
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_2d_function(func, bounds, shift, rotation=None, points=100, title="Function"):
    x = np.linspace(bounds[0][0], bounds[0][1], points)
    y = np.linspace(bounds[1][0], bounds[1][1], points)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros((points, points))
    for i in range(points):
        for j in range(points):
            point = np.array([X[i,j], Y[i,j]])
            Z[i,j] = func(point)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(Z, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], 
                   origin='lower', cmap='viridis')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    ax.plot(shift[0], shift[1], 'r*', markersize=15, label='Global Optimum')
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
    contours = ax.contour(X, Y, Z, colors='black', alpha=0.3, levels=15)
    ax.clabel(contours, inline=True, fontsize=8)
    
    plt.tight_layout()
    return fig

dim = 2
bounds = [(-5, 5)] * dim  
shift = generate_shift_vector(dim, bounds)
rotation = generate_rotation_matrix(dim)

functions = {
    'Shifted Sphere': lambda x: CECFunctions.shifted_sphere(x, shift),
    'Shifted Ackley': lambda x: CECFunctions.ackley(x, shift),
    'Shifted Rotated Elliptic': lambda x: CECFunctions.shifted_rotated_high_conditioned_elliptic(x, shift, rotation),
    'Shifted Rotated Griewank': lambda x: CECFunctions.shifted_rotated_griewank(x, shift, rotation)
}

for name, func in functions.items():
    fig = plot_2d_function(func, bounds, shift, rotation, points=100, title=name)
    plt.savefig(f"{name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

print(f"Shift vector used: {shift}")