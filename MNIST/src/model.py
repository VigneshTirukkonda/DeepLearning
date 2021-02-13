import config
import torch
import torch.nn as nn
import torch.nn.functional as F 


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Input = (batch, 1, 28, ou)  -- Channels 1 --> 10 = 28 - K + 1
            # CONV2D layer kernel_size = 29-10 = 19
        #  (batch, 3, 10, 10)-- Channels 3  --> out = N - k + 1
            # CONV2D layer kernel_size = 8
        #  (batch, 5, 3, 3)-- Channel 5

        # Flatten layer
        # FC layer 45 Units
        # FC layer 20 Units
        # Output = 10 way softmax


        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 3, kernel_size= 19)
        self.conv2 = nn.Conv2d(in_channels= 3, out_channels= 5, kernel_size= 8)
        self.FC1 = nn.Linear(in_features= 45, out_features= 20)
        self.FC2 = nn.Linear(in_features= 20, out_features= 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 45)
        x = F.relu(self.FC1(x))
        x = self.FC2(x)
        return x

