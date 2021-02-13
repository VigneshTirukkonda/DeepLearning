import config
import torchvision as tv
import torch.utils.data as data

transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,),(0.5,))
    ])


TRAIN_DS, VALID_DS = data.random_split(tv.datasets.MNIST(root= config.ROOT, download= True, transform= transform), [50000, 10000])

TRAIN_DL = data.DataLoader(dataset= TRAIN_DS, batch_size= config.BATCH_SIZE, shuffle= True)
VALID_DL = data.DataLoader(dataset= VALID_DS)


MNIST_TEST = tv.datasets.MNIST(root= config.ROOT, train= False, transform= transform)
TEST_DL = data.DataLoader(dataset= MNIST_TEST)

