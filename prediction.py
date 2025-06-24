import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from segmentation import get_character_images

model = load_model('cnn_model.keras')

def predict_image(img, label_map):
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred_class = np.argmax(pred)
    return label_map[int(pred_class)]

def get_label_map():
    class_names = sorted(os.listdir("datasets/cnn_dataset/"))
    return {k: v for k, v in enumerate(class_names)}

def get_plate_string(plate_image):
    label_map = get_label_map()
    char_images = get_character_images(plate_image, debug=True)
    predicted_string = ''.join([predict_image(image, label_map) for image in char_images])
    return predicted_string