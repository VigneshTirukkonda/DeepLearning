from flask import Flask, render_template, url_for, request
import json
from flask_jsglue import JSGlue
import config
import model
import torch

Model = model.CNN()
Model.load_state_dict(torch.load(config.MODEL_PATH + config.LATEST_MODEL))

app = Flask(__name__)
jsglue = JSGlue(app)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.unsqueeze(x, 0)),
        transforms.Normalize([0.5, ],[0.5, ])
        ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = Model.forward(tensor)
    return outputs.tolist()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # file = request.files['file']
        # img_bytes = file.read()
        img_bytes = json.loads(request.data['file'])
        # 2

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)