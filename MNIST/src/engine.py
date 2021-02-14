import config
import torch

criterion = torch.nn.CrossEntropyLoss()

def train_fn(data_loader, model, optmizer, device):
    model.train()
    total_loss = 0
    for inp, target in data_loader:
        optmizer.zero_grad()
        out = model(inp)
        loss = criterion(out, target)
        loss.backward()
        optmizer.step()
        total_loss += loss.item()

    return total_loss/len(data_loader)

def eval_fn(data_loader, model):
    model.eval()
    total_loss = 0
    for data in data_loader:
        loss = model(data)
        loss.backward()
        total_loss += loss.item()


    
