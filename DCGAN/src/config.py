import torch
DATAROOT = '../input'

# DEVICE AND GPU SPECS
NGPU = 1
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and NGPU > 0) else 'cpu')


# INPUT AND MODEL SPECS
IMAGE_SIZE = 28
NC = 1
NZ = 50
NGF = 32
NDF = 32


# TRAINING SPECS
NUM_EPOCHS = 5
BATCH_SIZE = 128
LR = 0.0002 # Page 3
BETA1 = 0.5 # Page 4