from tqdm import tqdm 

import config
import model
from dataset import *

import torch
import torchvision
from torch import nn
from torch import optim



def train(criterion, optimizer, models):
    # img_list = []
    # G_losses = []
    # D_losses = []
    # iters = 0
    # netG, netD = models

    # print('Starting training loop...')
    # for epoch in tqdm(range(config.NUM_EPOCHS)):
    #     netD.zero_grad()
    pass

if __name__=='__main__':
    fixed_noise = torch.randn(64, config.NZ, 1, 1, device=config.DEVICE)
    
    real_label = 1.
    fake_label = 0.

    netG = model.Generator(config.NGPU).to(config.DEVICE)
    netD = model.Discriminator(config.NGPU).to(config.DEVICE)

    if (config.DEVICE.type == 'cuda') and (config.NGPU > 1):
        netG = nn.DataParallel(netG, list(range(config.NGPU)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netG.apply(model.initializer)
    netD.apply(model.initializer)

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    optimizerD = optim.Adam(netD.parameters(), lr=config.LR, betas=(config.BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.LR, betas=(config.BETA1, 0.999))

    print('Starting training loop...')
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        
        for i, data in tqdm(enumerate(dloader, 0)):
            
            ## Update D Network: maximçize log((D(x))) + log(1 - D(G(z)))

            ## Train with all-real batch
            netD.zero_grad()

            real_cpu = data[0].to(config.DEVICE)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=config.DEVICE)
            
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)

            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(config.BATCH_SIZE, config.NZ, 1, 1, device=config.DEVICE)

            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)

            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_fake + errD_real

            optimizerD.step()

            ## UPDATE G NETWORK: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost

            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # output training stats:
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(dloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == config.NUM_EPOCHS-1) and (i == len(dloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake, padding=2, normalize=True))

            iters += 1