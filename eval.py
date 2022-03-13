from hmac import trans_36
import numpy as np
import os
import io

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str)

args = parser.parse_args()

image_path = args.image

image = keras.utils.load_img(image_path, target_size=(400, 400))
image_arr = keras.preprocessing.image.img_to_array(image)
image_arr_inp = np.array([image_arr])

model = keras.models.load_model("model")

test = model.predict(image_arr_inp)

pred = np.argmax(test)
pred_label = 'Male' if pred == 1 else 'Female'

plt.imshow(image_arr * 1/255)
plt.title(f'Prediction')
plt.axis('off')

plt.tight_layout()
plt.title(f'Label: {pred_label} - {test[0][pred] * 100:0.4f}%')

plt.show()