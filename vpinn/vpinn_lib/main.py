import torch 
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as leg
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.interpolate import UnivariateSpline

from training.model import create_model, create_optimizer
from training.metrics import compute_loss, compute_errors
from training.test_functions import generate_test_functions, FEM1D
from utils.problem import exact_u
from utils.visualization import plot_results, style_plot

N = 1000
nvals = 10
x = np.linspace(0, 1, N)

# FEM1D
stiff, mass = FEM1D(x)

# Vectores propios
vals, vecs = eigh(stiff[1:-1, 1:-1],
                  mass[1:-1, 1:-1],
                  subset_by_index=(0, nvals - 1))

vecs_comp = np.zeros((N, nvals))
vecs_comp[1:-1, :] = vecs

# Interpolación
x_eval = torch.linspace(0, 1, N).detach().numpy()
spline = UnivariateSpline(x, vecs_comp[:, 2], s=0, k=1)
v_eval = spline(x_eval)
v_eval = v_eval.reshape(N, 1)

def main():
    N_modes = 20
    n_pts = 1000
    iterations = 1000

    torch.manual_seed(123)
    model = create_model()    
    optimizer = create_optimizer(model)
    losses = []
    L2_errors = []    
    h1_norms = []      

    n = torch.linspace(1, N_modes, N_modes)     

    for i in range(iterations):        
        x = torch.linspace(0, np.pi, n_pts)        
        
        # Funciones de prueba
        v_test, v_dv = generate_test_functions(n_pts, vecs_comp)

        x = x.requires_grad_(True).view(n_pts, 1)   

        v = v_test
        dvs = v_dv
        
        optimizer.zero_grad()

        loss = compute_loss(model, x, v, dvs)        
        loss.backward(retain_graph=True)
        optimizer.step()                   

        losses.append(loss.item())
        L2_error, H1_norm = compute_errors(model, x, exact_u)
        L2_errors.append(L2_error)
        h1_norms.append(H1_norm)

        if (i+1) % 200 == 0:
            print(f"Loss at iteration {i+1}: {loss.item():.15f}")  
               
            x_test = torch.linspace(0, np.pi, n_pts).view(n_pts, 1)
            y_test = model(x_test).detach().numpy()    
            y_real = exact_u(x_test).detach().numpy() 
            style_plot(x_test, y_real, x_data=None, y_data=None, yh=y_test, i=i)   
            plt.savefig(f'plots/vpinn_{i+1}.pdf', dpi=300, bbox_inches='tight')    

    plot_results(losses, L2_errors, h1_norms)

if __name__ == "__main__":
    main()