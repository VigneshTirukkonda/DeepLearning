import config
import model
from dataset import *


import torch
import torchvision
from torch import nn

netG = model.Generator(config.NGPU).to(config.DEVICE)
netD = model.Discriminator(config.NGPU).to(config.DEVICE)

if (config.DEVICE.type == 'cuda') and (config.NGPU > 1):
    netG = nn.DataParallel(netG, list(range(config.NGPU)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

netG.apply(model.initializer)
netD.apply(model.initializer)

