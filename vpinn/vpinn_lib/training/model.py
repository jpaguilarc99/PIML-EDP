import torch
from torch import optim
from architectures.architectures import FCN 

def create_model():
    model = FCN(1, 1, 10, 4)
    return model

def create_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return optimizer