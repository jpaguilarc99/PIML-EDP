import torch

def compute_integral(y, x):    
    integral = torch.trapz(y, x, dim=0)
    return integral