import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 128, kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 2)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 16, kernel_size = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(18,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,1)

    def forward(self, x, parameter):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.cat((x,parameter), dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x