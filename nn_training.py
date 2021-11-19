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
loss_fn = nn.MSELoss()
label = torch.tensor([0., 1.])

for i in range(200):
    input_dx = torch.tensor(env.ee_vel)

    # s = torch.unsqueeze(torch.FloatTensor(env.ee_vel), 0)

    output_b = net.forward(input_dx)
    # output_b.numpy()

    # tt_list = []
    # tt_list.append(output_b[0][0])
    # print(output_b[0][0])
    # print(tt_list)

    dx_d = env.task_combination(output_b.detach().numpy().reshape(-1),
                                env.ee_pos_array.reshape(-1)).reshape(3, 1)

    loss = np.sum((dx_d - env.ee_vel_array) ** 2)
    print(loss)

    loss = torch.tensor(loss)
    # loss = loss_fn(output_b, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    xx = 1

    # loss_fn = env.step(output_b.detach().numpy().reshape(-1))
    # # loss = loss_nn(torch.tensor(env.loss_fn), torch.tensor(0.))
    # loss = torch.tensor(loss_fn)
    # print(loss)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()















