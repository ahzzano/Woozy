import numpy as np
import os
import io

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt


from PIL import Image

model = keras.models.load_model("model")

training_data = keras.utils.image_dataset_from_directory(
    directory='training_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=16,
    image_size=(400, 400),
    crop_to_aspect_ratio=True,
    color_mode='rgb'
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

file_writer = tf.summary.create_file_writer('./logs')

test_image = keras.utils.load_img('./test_data/16c8f0d22ca11d1c086a6243426a9211.png', target_size=(400, 400))
test_image_arr = keras.preprocessing.image.img_to_array(test_image)

def test_image_identification(epoch, logs):
    test = model.predict(np.array([test_image_arr]))

    pred = np.argmax(test)
    pred_label = 'Male' if pred == 1 else 'Female'

    figure = plt.figure(figsize=(8, 8))

    plt.imshow(test_image_arr * 1/255)
    plt.title(f'Prediction')
    plt.axis('off')

    plt.tight_layout()
    plt.title(f'Label: {pred_label} - {test[0][pred]:0.4f}%')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    with file_writer.as_default():
        image = tf.image.decode_png(buf.getvalue(), channels=3)
        image = tf.expand_dims(image, 0)
        tf.summary.image('Prediction', image, step=epoch)


history = model.fit(training_data, epochs=20, callbacks=[
    tensorboard_callback,
    keras.callbacks.LambdaCallback(on_epoch_end=test_image_identification)
])

model.save("model")