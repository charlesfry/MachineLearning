import os
import numpy as np
import math
import torch
import random

def seed_everything(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_everything(seed:=69)

# create input/output
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# randomly init weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6

def get_loss(y_pred, y):
    return np.square(y_pred - y).sum()

for t in range(steps:=2000):
    # forward pass: compute y-pred
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # compute and show loss
    loss = get_loss(y_pred, y)
    if t % 100 == 99:
        print(f'step {t}: loss {loss}')

    # backprop to compute gradients w.r.t. loss
    grad_y_pred = 2 * (y_pred - y)
    grad_a = grad_y_pred.sum() # * 1
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'\nResult: y = {a} + {b}x + {c}x^2 + {d}x^3')
y_pred = a + b * x + c * x ** 2 + d * x ** 3
print(f'Loss: {get_loss(y_pred, y)}\n--------------------\n')

dtype = torch.float
# device = torch.device('cpu')
device = torch.device('cuda:0')

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# randomly init weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # forward pass
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    loss = torch.square(y_pred - y).sum().item()
    if t * 100 == 99:
        print(t, loss)

    # backprop
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

y_pred = a + b * x + c * x ** 2 + d * x ** 3
print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
print(loss:=torch.square(y_pred - y).sum().item())
