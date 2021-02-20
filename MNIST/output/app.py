from flask import Flask, render_template, url_for, request
import json
from flask_jsglue import JSGlue
import config
import model
import torch
from torchvision import transforms
import requests

# img_transform = transforms.Compose([
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: torch.unsqueeze(x, 0)),
#     transforms.Normalize([0.5, ],[0.5, ])
#     ])


# Model = model.CNNJS(img_transform)
# Model.load_state_dict(torch.load(config.MODEL_PATH + config.LATEST_MODEL))

app = Flask(__name__)
jsglue = JSGlue(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/testpost', methods=['POST'])
def testpost2(): 
    if request.method == 'POST':
        url = request.json['url']
        response = requests.get(url)
        file = open('test.png',"wb")
        file.write(response.content)
        file.close()
        print('Got img')
    return 'OK'

if __name__ == '__main__':
    app.run(debug=True)