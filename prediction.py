import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from segmentation import get_character_images
from tools import get_label_map

class Prediction:
    def __init__(self, model_path, debug):
        self.model = load_model(model_path)
        self.label_map = get_label_map()
        self.debug = debug

    def _predict_image(self, img):
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = self.model.predict(img_array, verbose=0)
        pred_class = np.argmax(pred)
        return self.label_map[int(pred_class)]

    def get_plate_string(self, plate_image):
        char_images = get_character_images(plate_image, self.debug)
        predicted_string = ''.join([self._predict_image(image) for image in char_images])
        return predicted_string