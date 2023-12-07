import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class Model(torch.nn.Module):
    def __init__(self, neurons, n_layers, activation=torch.tanh):
        super(Model, self).__init__()
        self.activation = activation
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(1, neurons))
        for _ in range(n_layers-2):
            self.layers.append(torch.nn.Linear(neurons, neurons))
        self.layers.append(torch.nn.Linear(neurons, 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

def f_rhs(x):
    return -4*torch.sin(2 * x)

def loss_fn(u_model, x, f):
    u = u_model(x)
    du = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    ddu = grad(du, x, grad_outputs=torch.ones_like(du), create_graph=True)[0]
    error_PDE = torch.mean((ddu - f(x))**2)
    bc = u_model(torch.tensor([np.pi]))**2 + u_model(torch.tensor([0.]))**2
    return error_PDE + bc[0]

def train(model, optimizer, loss_fn, f, n_pts, iterations):
    losses = []
    for iteration in range(iterations):  


        optimizer.zero_grad()
        x = torch.FloatTensor(n_pts,1).uniform_(0, np.pi).requires_grad_(True)
        loss = loss_fn(model, x, f)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if iteration % 1000 == 0:
            print(f'Iteration {iteration}, Loss {loss.item()}')
    return losses

def exact_u(x):
    return torch.sin(2 * x)

nn = 10
nl = 4
n_pts = 1000
iterations = 100

model = Model(neurons=nn, n_layers=nl)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = train(model, optimizer, loss_fn, f_rhs, n_pts, iterations)

xlist = np.array([np.pi/1000 * i for i in range(1000)])
xlist_torch = torch.tensor(xlist, dtype=torch.float32).view(-1, 1)

plt.plot(xlist, model(xlist_torch).detach().numpy(), color='b')
plt.plot(xlist, exact_u(xlist_torch).detach().numpy(), color='m')
plt.legend(['u_approx', 'u_exact'])
plt.grid(which = 'both', axis = 'both', linestyle = ':', color = 'gray')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
plt.plot(losses, color='r')
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend(['loss'])
ax.grid(which = 'major', axis = 'both', linestyle = ':', color = 'gray')
plt.tight_layout()
plt.show()