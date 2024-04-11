from flask import Flask, request, jsonify
from keras.models import model_from_json
import numpy as np
import cv2
import base64
from flask_cors import CORS
import pandas as pd
import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import load_model

app = Flask(__name__)
CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


def load_rgba_image(image_data):
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if image.shape[2] != 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image

# Modeli yükleyin

def load_my_model():
    loaded_model = load_model('vgg19_cnn.h5')
    return loaded_model


emotion_model = load_my_model()

@app.route('/print_something', methods=['GET'])
def printSomething():
    print("Request received")
    return "Hello World"

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    print("Request received")
    data = request.get_json()
    image_base64 = data['image']
    # Base64 string'i RGBA görüntüye dönüştür
    img = load_rgba_image(image_base64.split(',')[1])

    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))
    rgb_frame = np.dstack([resized_frame]*3)

    # Modelin beklediği şekle getir (1, 48, 48, 3)
    processed_frame = np.expand_dims(rgb_frame, 0)

    # Save the processed image
    # cv2.imwrite('processed_image.jpg', img)

    emotion_prediction = emotion_model.predict(processed_frame)
    maxindex = int(np.argmax(emotion_prediction))
    print(emotion_prediction)
    probabilities = {emotion_dict[i]: float(emotion_prediction[0][i]) for i in range(len(emotion_dict))}

    return jsonify({"emotion": emotion_dict[maxindex], "probabilities": probabilities})

if __name__ == '__main__':
    app.run(debug=False, port=5500)
