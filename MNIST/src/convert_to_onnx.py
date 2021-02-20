import torch 
import model
import config
from torchvision import transforms

def main():

    Model = model.CNN()
    Model.load_state_dict(torch.load(config.MODEL_PATH + config.LATEST_MODEL))
    Model.eval()
    dummy_input = torch.zeros(280 * 280 * 4)
    torch.onnx.export(Model, dummy_input, '../output/static' + '/'+'.'.join(config.LATEST_MODEL.split('.')[0:2])+'.onnx', verbose= True)
    print('Model Created : ../output/static' + '/'+'.'.join(config.LATEST_MODEL.split('.')[0:2])+'.onnx')

if __name__ == '__main__':
    main()
