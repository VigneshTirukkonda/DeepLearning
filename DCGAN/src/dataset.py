import config

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torchvision import datasets

dset = datasets.MNIST(root= config.DATAROOT)
dloader = data.DataLoader(dset, batch_size=config.BATCH_SIZE)


