import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias)

        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.Tensor(1, out_features))

        self.register_buffer(name='epsilon_weight', tensor=torch.zeros((out_features, in_features)))
        self.register_buffer(name='epsilon_bias', tensor=torch.zeros((1, out_features)))

        self.reset_parameters()

    # def reset_parameters(self):
    #     if hasattr(self, 'sigma_weight'):
    #         nn.init.xavier_uniform_(self.weight)
    #         nn.init.constant(self.sigma_weight, self.sigma_init)
    #         nn.init.constant(self.sigma_bias, self.sigma_init)

    def forward(self, x):

        out = F.linear(x,
                       weight=self.weight + self.sigma_weight * Variable(self.epsilon_weight),
                       bias=self.bias + self.sigma_bias * Variable(self.epsilon_bias))

        return out

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)

# approximate Q function using Neural Network
# state is input and Q Value of each action is output of network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            NoisyLinear(state_size, 24),
            nn.ReLU(),
            NoisyLinear(24, 24),
            nn.ReLU(),
            NoisyLinear(24, action_size)
        )

    def forward(self, x):
        return self.fc(x)

class DoubleDuelingDQN(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)

        self.VA = nn.Linear(16, action_size)
        self.VS = nn.Linear(16, 1)

    def forward(self, x):

        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        z3 = F.relu(self.fc3(z2))

        va = self.VA(z3)
        vs = self.VS(z3)

        # the dimensions of va and vs don't match
        avg_va = torch.mean(va, dim=1, keepdim=True)

        out = vs + (va - avg_va)

        return out
