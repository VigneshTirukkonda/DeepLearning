import config
import dataset 
import engine
import model
import torch
import torch.optim as optim
import datetime
from tqdm import tqdm

if __name__ == "__main__":
    
    Model = model.CNN()
    Optmizer = optim.SGD(Model.parameters(), lr= config.LR, momentum= config.MOMENTUM)
    # Model.load_state_dict(torch.load(config.MODEL_PATH))

    for epoch in tqdm(range(config.EPOCHS)):
        print('Epoch {} Loss: {}'.format(epoch, engine.train_fn(dataset.TRAIN_DL, Model, Optmizer, 'CPU')))

    torch.save(Model.state_dict(), config.MODEL_PATH + '/CNN-' + str(datetime.datetime.today()) + '.pt')

    correct = 0
    for img, target in tqdm(dataset.TEST_DL):
        Model.eval()
        pred = torch.unsqueeze(torch.argmax(Model(img)), 0)
        correct += 1 if torch.equal(pred, target) else 0

    print('Test accuracy: {} % '.format(correct/len(dataset.TEST_DL)*100))
