import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, task_num):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, task_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.unsqueeze(x, 0)
        x = nn.Softmax(dim=1)(x)
        return x


if __name__ == "__main__":
    input = torch.rand(3,)
    net = Net(2)
    output = net.forward(input)
    print('output')
    print(output, output.shape)






























