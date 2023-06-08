import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Others
import numpy as np
from util import base64_to_pil


# Инициализируем Flask app
app = Flask(__name__)


# Можно использовать любые обученные модели
# https://keras.io/applications/
# https://www.tensorflow.org/api_docs/python/tf/keras/applications

#from tensorflow.keras.applications.efficientnet import EfficientNetB7
#model = EfficientNetB7(weights='imagenet')

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:5000/')


# Сохраненная модель
MODEL_PATH = 'models/your_model.h5'

# Можно загрузить свою модель
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


def model_predict(img, model):

    img = img.resize((224, 224))

    # Подготовка изображения
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Скорретировать ввод под используемую модель
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Главная страница
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Получить изображение из POST
        img = base64_to_pil(request.json)


        # Делаем прогноз
        preds = model_predict(img, model)

        # Демонстрируем результат
        pred_proba = "{:.3f}".format(np.amax(preds))    # Вероятность
        pred_class = decode_predictions(preds, top=1)   # Класс картинки


        result = str(pred_class[0][0][1])
        result = result.replace('_', ' ').capitalize()

        # Сохраняем картинку to ./uploads
        img.save("./uploads/" +result +".png")
        
        # Сериализация данных и ввывод лога
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Запуск сервера
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
