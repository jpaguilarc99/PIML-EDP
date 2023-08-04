import torch
import torch.nn as nn  
import torch.optim as optim
import numpy as np  
import matplotlib.pyplot as plt  

def encode_inputs(x, t):
    x_normalized = (x - x.min()) / (x.max() - x.min())
    t_normalized = (t - t.min()) / (t.max() - t.min())

    inputs = torch.cat((x_normalized, t_normalized), dim=1)
    return inputs

def true_solution(x, t):
    return torch.sin(np.pi * x) * np.exp(-np.pi**2 * t)  

"""
def pde_loss(model, x_pde, t_pde):
    x_pde.requires_grad = True
    t_pde.requires_grad = True
    u_pde = model(x_pde, t_pde)
    u_t = torch.autograd.grad(u_pde, t_pde, torch.ones_like(u_pde), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pde, x_pde, torch.ones_like(u_pde), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, torch.ones_like(u_x), create_graph=True)[0]
    pde_residual = u_t - u_xx
    return torch.mean(pde_residual**2)

def boundary_loss(model, x_boundary):
    u_boundary = model(x_boundary, torch.zeros_like(x_boundary))  # t=0 en las condiciones de frontera
    return torch.mean(u_boundary**2)
"""  

class TransformerEDP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim):
        super(TransformerEDP, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)  
        # x = x.permute(1, 0, 2)
        output = self.transformer_encoder(x)  
        # output = output.permute(1, 0, 2)
        output = self.decoder(output)
        return output 

def train_model(model, inputs, true_output, num_epochs=2000, lr=0.001):
    criterion = nn.MSELoss()    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predicted_output = model(inputs)
        loss = criterion(predicted_output, true_output)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Definir la malla espacial y temporal
num_points_x = 100
num_points_t = 100
x = torch.linspace(0, 1, num_points_x).view(-1, 1)
t = torch.linspace(0, 1, num_points_t).view(-1, 1)

inputs = encode_inputs(x, t)

true_output = true_solution(x, t)

model = TransformerEDP(input_dim=2, output_dim=1, num_layers=4, hidden_dim=64)
train_model(model, inputs, true_output)

predicted_output = model(inputs)

plt.plot(x.numpy(), true_output.numpy(), label='True Solution')
plt.plot(x.numpy(), predicted_output.detach().numpy(), label='Predicted Solution', lw=0, marker="o")
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('True vs Predicted Solutions')
plt.show()