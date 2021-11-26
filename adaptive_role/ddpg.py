#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.98     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

###############################  DDPG  ####################################


class ANet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        self.a_dim = a_dim
        super(ANet,self).__init__()
        self.fc1 = nn.Linear(s_dim,30)
        self.out = nn.Linear(30,a_dim)

    def forward(self, x, noise_scale):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        y = torch.tensor(np.random.normal(0., noise_scale, (1, self.a_dim)).tolist())
        z = x + y
        z = nn.Softmax(dim=1)(z)
        return z


class CNet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(CNet,self).__init__()
        self.fcs = nn.Linear(s_dim,30)
        self.fca = nn.Linear(a_dim,30)
        self.out = nn.Linear(30, 1)

    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        net = self.out(net)
        return net


class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.a_dim, self.s_dim = a_dim, s_dim
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim,a_dim)
        self.Actor_target = ANet(s_dim,a_dim)
        self.Critic_eval = CNet(s_dim,a_dim)
        self.Critic_target = CNet(s_dim,a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s, noise_scale):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_eval(s, noise_scale)[0].detach() # ae（s）

    def learn(self):
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement
        #self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(bs, 0.)
        q = self.Critic_eval(bs,a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_, 0.)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_,a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br+GAMMA*q_  # q_target = 负的
        #print(q_target)
        q_v = self.Critic_eval(bs,ba)
        #print(q_v)
        td_error = self.loss_td(q_target,q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        #print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save_model(self, output):
        torch.save(
            self.Actor_eval.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.Critic_eval.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def load_weights(self, output):
        if output is None:
            return

        self.Actor_eval.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.Critic_eval.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def eval(self):
        self.Actor_eval.eval()
        self.Actor_target.eval()
        self.Critic_eval.eval()
        self.Critic_target.eval()


if __name__ == "__main__":
    # input = torch.rand(3, )
    # anet = ANet(3, 2)
    # output = anet.forward(input, 0.1)
    # print('output')
    # print(output, output.shape)

    ddpg = DDPG(2, 3)
    s = np.array([1., 2., 3.])
    output_1 = ddpg.choose_action(s, 0.1)
    # print('output_1')
    # print(output_1, output_1.shape)
    # print('^^^')
    # print(output_1.numpy(), output_1.numpy().shape)












