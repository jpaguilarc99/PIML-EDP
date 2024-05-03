import torch
import numpy as np
from scipy.interpolate import UnivariateSpline

def FEM1D(coords):
    N = len(coords)
    stiff_loc = np.array([[2.0, -2.0],
                          [-2.0, 2.0]])
    mass_loc = np.array([[2/3, 1/3],
                         [1/3, 2/3]])
    eles = [np.array([cont, cont + 1]) for cont in range(0, N - 1)]
    stiff = np.zeros((N, N))
    mass = np.zeros((N, N))
    for ele in eles:  
        jaco = coords[ele[1]] - coords[ele[0]]
        for cont1, row in enumerate(ele):
            for cont2, col in enumerate(ele):
                stiff[row, col] = stiff[row, col] +  stiff_loc[cont1, cont2]/jaco
                mass[row, col] = mass[row, col] +  jaco*mass_loc[cont1, cont2]
    return stiff, mass

def generate_test_functions(n_pts, vecs_comp):
    v_test = torch.zeros(n_pts, vecs_comp.shape[1])
    v_dv = torch.zeros(n_pts, vecs_comp.shape[1])
    x = torch.linspace(0, np.pi, n_pts)
    for j in range(vecs_comp.shape[1]):
        spline_mode = UnivariateSpline(x, vecs_comp[:, j], s=0, k=1)
        v_mode = spline_mode(x)  
        v_mode = v_mode.reshape(n_pts, 1)              
        v_test[:, j] = torch.tensor(v_mode).squeeze()
        v_dv[:, j] = torch.tensor(spline_mode.derivative()(x)).squeeze()  
    return v_test, v_dv 