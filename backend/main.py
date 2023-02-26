from PIL import Image
from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
import keras.utils as image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify
from skimage import io
from skimage.transform import resize
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def read_img_from_path(img_path):
    rgb = io.imread(img_path)
    resized_image = resize(rgb, (180, 180))
    rescaled_image = 255 * resized_image
    final_image = rescaled_image.astype(np.uint8)
    img = Image.fromarray(final_image, 'RGB')
    return img


def getprediction(img):
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)

    model = load_model('./final_model_weights.hdf5')
    predict = model.predict('https://images.app.goo.gl/s1NEp8sGpB1623xQ7')
    classes = np.argmax(predict, axis=1)

    if classes == 1:
        return "Recycle"
    else:
        return "Organic"


# img_url = request.args.get('img', './bottle.jpg')
# img = read_img_from_path('https://i.stack.imgur.com/QupKb.png')
# ans = getprediction(img)
# print(ans)

@app.route('/')
def home():
    return "<h1>Welcome to ROBbish</h1>"


@app.route('/predict/', methods=['GET'])
def predict():
    img_url = request.args.get(
        'img', default='https://i.stack.imgur.com/QupKb.png')

    img = read_img_from_path(img_url)
    ans = getprediction(img)
    return jsonify({"ans": ans})
