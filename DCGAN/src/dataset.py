import config

import numpy as np 
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from torchvision import utils

dset = datasets.MNIST(root= config.DATAROOT
                    ,transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]))
dloader = data.DataLoader(dset, batch_size=config.BATCH_SIZE)

# if __name__ == '__main__':
#     batch, _ = next(iter(dloader))
#     plt.figure(figsize=(8,8))
#     plt.axis('off')
#     plt.title('training examples')
#     plt.imshow(np.transpose(utils.make_grid(batch.to(config.DEVICE)[:64], padding=2, normalize= True).cpu(), (1,2,0)))
#     plt.show()