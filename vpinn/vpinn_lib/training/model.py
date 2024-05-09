import torch
from torch import optim
from architectures.architectures import FCN, CNNPDE, RNNPDE, TransformerEDP 

def create_model(model_name):
    if model_name == "FFNN":
        return FCN(1, 1, 10, 4)
    if model_name == "CNN":
        return CNNPDE()
    if model_name == "RNN":
        return RNNPDE()
    if model_name == "Transformer":
        return TransformerEDP(input_dim=1, output_dim=1, num_layers=2, hidden_dim=10)
    return FCN(1, 1, 10, 4)

def create_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return optimizer