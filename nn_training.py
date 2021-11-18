from hri_env_v1 import HriEnv_v1
from nn_belief import Net
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

env = HriEnv_v1()
env.reset()
net = Net(env.task_num)
learning_rate = 1e-3
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
# loss_nn = nn.MSELoss()

for i in range(200):
    input_dx = torch.tensor(env.ee_vel)
    output_b = net.forward(input_dx)
    loss_fn = env.step(output_b.detach().numpy().reshape(-1))
    # loss = loss_nn(torch.tensor(env.loss_fn), torch.tensor(0.))
    loss = torch.tensor(loss_fn)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()















