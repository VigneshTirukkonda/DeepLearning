import config
import torch
import torchvision
from torch import nn 


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(config.NC, config.NDF, kernel_size=4, stride=2, padding=1),    
            nn.LeakyReLU(0.2, inplace=True),

            # stage shape (ndf) x 14 x 14
            nn.Conv2d(config.NDF, config.NDF * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # stage shape (ndf x 2) x 7 x 7
            nn.Conv2d(config.NDF * 2, config.NDF * 4, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),

            # stage shape (ndf x 4) x 3 x 3
            nn.Conv2d(config.NDF * 4, 1, kernel_size= 3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Sigmoid()            
        )
    
    def forward(self, x):
        return self.main(x)
    
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()   

        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input Z
            nn.ConvTranspose2d(config.NZ, config.NGF * 4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(config.NGF * 4),
            nn.ReLU(inplace=True),

            # stage shape (ngf x 4) x 3 x 3
            nn.ConvTranspose2d(config.NGF * 4, config.NGF * 2, kernel_size=3, stride= 2, bias= False),
            nn.BatchNorm2d(config.NGF * 2),
            nn.ReLU(inplace=True),

            # stage shape (ngf x 2) x 7 x 7
            nn.ConvTranspose2d(config.NGF * 2, config.NGF, kernel_size=4, stride= 2, padding=1, bias= False),
            nn.BatchNorm2d(config.NGF),
            nn.ReLU(inplace=True),

            # stage shape (ngf) x 14 x 14
            nn.ConvTranspose2d(config.NGF, config.NC, kernel_size=4, stride= 2, padding=1, bias= False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)

def initializer(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1, 0.02)
        nn.init.constant_(m.bias.data, 0)
         
        
