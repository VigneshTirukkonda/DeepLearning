import datetime
import matplotlib.pyplot as plt
import model
import dataset
import config
from tqdm import tqdm as tqdm
import numpy as np
import engine
import torch.optim as optim
from torchvision import transforms as T 
from torchvision import datasets as D
from torch.utils import data 
from torchvision import utils as U

trans = T.Compose({
    T.RandomRotation(degrees=20),
    T.ToTensor(),
    T.Normalize((0.5,),(0.5,))
})

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1 ,2 ,0)))

if __name__ == '__main__':
    DS = D.MNIST(root= config.ROOT, train= True, transform= trans)
    DL = data.DataLoader(DS, batch_size= config.BATCH_SIZE)

    Model = model.CNNtrain()
    Optmizer = optim.SGD(Model.parameters(), lr= config.LR, momentum= config.MOMENTUM)
    # Model.load_state_dict(torch.load(config.MODEL_PATH))

    for epoch in tqdm(range(config.EPOCHS)):
        print('Epoch {} Loss: {}'.format(epoch, engine.train_fn(DL, Model, Optmizer, 'CPU')))

    torch.save(Model.state_dict(), config.MODEL_PATH + '/CNN-' + str(datetime.datetime.today()) + '.pt')

    correct = 0
    for img, target in tqdm(dataset.TEST_DL):
        Model.eval()
        pred = torch.unsqueeze(torch.argmax(Model(img)), 0)
        correct += 1 if torch.equal(pred, target) else 0

    print('Test accuracy: {} % '.format(correct/len(dataset.TEST_DL)*100))

    
    # imgs, _ = iter(DL).next()
    # img_grid = U.make_grid(imgs)
    # show(img_grid)




