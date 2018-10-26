import torch.nn as nn


class MyDQN(nn.Module):
    def __init__(self, nr_actions):
        super(MyDQN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0),
            nn.ReLU())

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=0),
            nn.ReLU())

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU())
        self.head = nn.Linear(in_features=512, out_features=nr_actions)

    def forward(self, x):

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.head(x)
