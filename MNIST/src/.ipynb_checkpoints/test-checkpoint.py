import gzip
import numpy as np
import torch 
import dataset
from torch.utils.data import Dataset
import config
import os
import pandas as pd
import model
from PIL import Image


def process_img(IMG_FILE):
    img = Image.open(config.TEST_PATH + IMG_FILE).convert('L')
    img.thumbnail(config.IMG_DIM, Image.ANTIALIAS)
    return np.array(img)

class Testset(Dataset):

    def __init__(self, imgs, targets, transform = None):
        
        self.imgs = imgs
        self.targets = targets 
        self.transform = transform       

    def __getitem__(self, index):
        
        if self.transform:
            imgs[index] = self.transform(imgs[index])

        return (self.imgs[index], self.targets[index])

    def __len__(self):
        
        return len(self.imgs)


if __name__ == '__main__':
    impath_list = []
    for root, _, impath in os.walk(config.TEST_PATH):
        impath_list += impath

    data = []
    for impath in impath_list:
        data.append(process_img('/'+impath))
    targets = [8, 4, 3, 1]

    DS = Testset(data, targets, transform= dataset.transform)
    DL = torch.utils.data.DataLoader(DS)

    Model = model.CNN()
    Model.load_state_dict(torch.load(config.MODEL_PATH + '/CNN.pt12FebCNN'))
    # Model.load_state_dict(torch.load(config.MODEL_PATH + 'CNN.pt'))




