import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class BranchNet(nn.Module):
    def __init__(self):
        super(BranchNet, self).__init__()
        self.layer = nn.Linear(1, 20)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.layer(x))

class TrunkNet(nn.Module):
    def __init__(self):
        super(TrunkNet, self).__init__()
        self.layer1 = nn.Linear(1, 20)
        self.layer2 = nn.Linear(20, 20)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.layer2(x)
        return self.tanh(x)

class DeepONet(nn.Module):
    def __init__(self, branch, trunk):
        super(DeepONet, self).__init__()
        self.branch = branch
        self.trunk = trunk

    def forward(self, x):
        B = self.branch(x)
        T = self.trunk(x)
        return torch.sum(B * T, dim=1, keepdim=True)

def pinn_loss(model, x):
    u = model(x)
    u_xx = torch.autograd.grad(u.sum(), x, create_graph=True, allow_unused=True)[0]
    f = -torch.sin(np.pi * x)
    return torch.mean((u_xx - f)**2)

branch = BranchNet()
trunk = TrunkNet()
model = DeepONet(branch, trunk)

optimizer = optim.Adam(model.parameters(), lr=0.01)

x = torch.linspace(0, 1, 1000).reshape(-1, 1).requires_grad_(True)

import matplotlib.pyplot as plt

losses = []
for epoch in range(20000):
    optimizer.zero_grad()
    loss = pinn_loss(model, x)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f'Epoch {epoch}, Loss: {loss.item()}')

plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

u_real = torch.sin(np.pi * x) / np.pi**2

u = model(x).detach().numpy()
u_real = u_real.detach().numpy()
plt.figure()
plt.plot(x.detach().numpy(), u, label='Model output')
plt.plot(x.detach().numpy(), u_real, label='Real solution')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()