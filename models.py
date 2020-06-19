import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, obs_shape, num_actions, learning_rate=0.0001):
        super(Model, self).__init__()
        assert len(
            obs_shape) == 1, "This network only works for flat observations"
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.net = nn.Sequential(
            torch.nn.Linear(obs_shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions),
        )
        self.opt = optim.Adam(self.net.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.net(x)

class ConvModel(nn.Module):
    def __init__(self, obs_shape, num_actions, learning_rate=0.0001):
        assert len(obs_shape) == 3, "This network only works for rgb image observations"
        super(ConvModel, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.conv_net = nn.Sequential(
            torch.nn.Conv2d(4, 16, (8,8), stride=(4, 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, (4, 4), stride=(2, 2)),
            torch.nn.ReLU(),
        )

        with torch.no_grad():
            #computes the size of the fully connected layer...because torch doesn't do it for you
            dummy = torch.zeros((1, *obs_shape))
            x = self.conv_net(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]

        self.fc_net = nn.Sequential(
            torch.nn.Linear(fc_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions),
        )

        self.opt = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        conv_latent = self.conv_net(x/255.0) # shape: (N, )
        # conv_latent is flattened using .view()
        return self.fc_net(conv_latent.view((conv_latent.shape[0], -1)) 
)


# if __name__ == '__main__':
#     # m = ConvModel((4, 84, 84), 4)
#     # t = torch.zeros((1,4,84,84))
#     # print(m.forward(t))
