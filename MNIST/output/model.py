import config
import torch
import torch.nn as nn
import torch.nn.functional as F 


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 3, kernel_size= 19)
        self.conv2 = nn.Conv2d(in_channels= 3, out_channels= 5, kernel_size= 8)
        self.FC1 = nn.Linear(in_features= 45, out_features= 20)
        self.FC2 = nn.Linear(in_features= 20, out_features= 10)

    def forward(self, x):

        x = x.reshape(280, 280, 4)
        x = torch.narrow(x, dim=2, start=3, length=1)
        x = x.reshape(1, 1, 280, 280)
        x = F.avg_pool2d(x, 10, stride=10)
        x = x/255
        x = (x-0.5)/0.5

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 45)
        x = F.relu(self.FC1(x))
        x = self.FC2(x)
        x = F.softmax(x)
        return x


