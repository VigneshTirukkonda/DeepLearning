import gzip
import numpy as np
import torch 
import dataset
from torch.utils.data import Dataset
import config
import os
import pandas as pd
import model
from tqdm import tqdm
from PIL import Image


class Testset(Dataset):

    def __init__(self, filepath, transform = None):
        self.filepath_df = pd.read_csv(filepath)
        self.transform = transform       

    def __getitem__(self, index):
        if self.transform:
            img = self.process_img(self.filepath_df.iloc[index, 0])
            img = self.transform(img)
        return (img, self.filepath_df.iloc[index, 1])

    def __len__(self):
        return len(self.filepath_df)

    def process_img(self, IMG_FILE):
        img = Image.open(config.TEST_PATH + IMG_FILE).convert('L')
        img.thumbnail(config.IMG_DIM, Image.ANTIALIAS)
        return np.array(img)


if __name__ == '__main__':
    DS = Testset(config.TEST_PATH + '/testset.csv', transform= dataset.transform)
    DL = torch.utils.data.DataLoader(DS)

    Model = model.CNN()
    Model.load_state_dict(torch.load(config.MODEL_PATH + '/CNN-2021-02-14 13:45:49.382405.pt'))

    correct = 0
    for img, target in tqdm(DL):
        Model.eval()
        pred = torch.unsqueeze(torch.argmax(Model(img)), 0)
        correct += 1 if torch.equal(pred, target) else 0

    print('Test accuracy: {} % '.format(correct/len(DL)*100))


