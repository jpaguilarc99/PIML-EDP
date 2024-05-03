import torch
from utils.problem import exact_u_prime
from utils.methods import compute_integral
from utils.problem import f_rhs, exact_u_prime

def compute_loss(model, x, v, v_dv):
    u = model(x)
    du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]     
    
    dvs = v_dv
        
    f_rhs_val = f_rhs(x)    
    #integral_left = compute_integral(torch.einsum('jl,kjl->jk', du, dvs),x)    BUENO SINS FUNCS
    integral_left = compute_integral(torch.einsum('ij,ij->ij', du, dvs), x) # PRUEBA HELMHOLTZ
    integral_right = compute_integral(torch.einsum('jl,jk->jk', f_rhs_val, v), x)

    error_PDE = (integral_left - integral_right)**2    
    loss = error_PDE.sum()    

    return loss

def compute_errors(model, x, exact_u):
    exact_function = exact_u(x)
    predicted_function = model(x)

    error_between_functions = exact_function - predicted_function
    error_squared = error_between_functions**2
    L2_error = compute_integral(error_squared, x)

    predicted_derivative = torch.autograd.grad(predicted_function, x, torch.ones_like(predicted_function), create_graph=True, retain_graph=True)[0]
    exact_derivative = exact_u_prime(x)
    error_between_derivatives = exact_derivative - predicted_derivative
    error_derivative_squared = error_between_derivatives**2

    H1_error = compute_integral(error_squared + error_derivative_squared, x)
    
    exact_function_squared = exact_function**2
    exact_function_norm = compute_integral(exact_function_squared, x)

    relative_H1_error = H1_error / exact_function_norm

    return torch.sqrt(L2_error).item(), torch.sqrt(relative_H1_error).item()