import io
import model
import torch
import config
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, url_for, request, jsonify




def transform_img(image_bytes):
    transform = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.unsqueeze(x, 0)),
        transforms.Normalize((0.5,), (0.5))
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    return transform(image)

def get_prediction(image_bytes):
    tensor = transform_img(image_bytes)
    outputs = Model.forward(tensor)
    return outputs.tolist()


app = Flask(__name__)
Model = model.CNN()
Model.load_state_dict(torch.load(config.MODEL_PATH + config.LATEST_MODEL))
Model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        predictions = get_prediction(img_bytes)
    return jsonify({'class_probs': predictions})

if __name__ == "__main__":
    app.run(debug=True)