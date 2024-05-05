import torch 

def f_rhs(x):
  return 0.*x-2. 

def exact_u(x):
  return x*(x-torch.pi) 

def exact_u_prime(x):
    return 2*x - torch.pi

def compute_integral(y, x):    
    integral = torch.trapz(y, x, dim=0)
    return integral