import gzip
import numpy as np
import torch 
import config
from PIL import Image


def read_MNISTdata(MNIST_PATH='../Input/MNIST/raw', IMG_FILE='/t10k-images-idx3-ubyte.gz', TARGET_FILE='/t10k-labels-idx1-ubyte.gz', image_size = 28, num_images = 5):

    f = gzip.open(MNIST_PATH + IMG_FILE,'r')
    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    f.close()

    f = gzip.open(MNIST_PATH + TARGET_FILE,'r')
    f.read(8)
    buf = f.read(num_images)
    targets = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    f.close()

    return data, targets

def process_img(IMG_FILE):
    img = Image.open(config.TEST_PATH + IMG_FILE).convert('L')
    img.thumbnail(config.IMG_DIM, Image.ANTIALIAS)
    return np.array(img)



class Testset():
    def __init__(self, imgs, targets):
        self.imgs = imgs
        self.targets = targets        

    def __getitem__(self, index):
        return (self.imgs[index], self.targets[index])

    def __len__(self):
        return len(self.imgs)

# if __name__ == '__main__':
#     data, target = read_MNISTdata()
#     dataset = Testset(data, targets)
#     dataloader = torch.utils.data.DataLoader(dataset)

