import torch 
import model
import config

def main():
    Model = model.CNN()
    Model.load_state_dict(torch.load(config.MODEL_PATH + config.LATEST_MODEL))
    Model.eval()
    dummy_input = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(Model, dummy_input, config.MODEL_PATH + '/'+'.'.join(config.LATEST_MODEL.split('.')[0:2])+'.onnx', verbose= True)

if __name__ == '__main__':
    main()
