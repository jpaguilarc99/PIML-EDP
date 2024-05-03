import torch 

def f_rhs(x):
  return 0.*x-2. #4*torch.sin(2 * x) # #

def exact_u(x):
  return x*(x-torch.pi) # #torch.sin(2 * x) #

def exact_u_prime(x):
    return 2*x - torch.pi