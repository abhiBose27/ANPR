import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from tools import get_label_map


class OCREngine:
    def __init__(self, model_path, prediction_threshold=0.8):
        self.model = load_model(model_path)
        self.label_map = get_label_map()
        self.prediction_threshold = prediction_threshold

    def run_batch_ocr(self, char_images):
        total_confidence = 0
        predicted_string = ""
        char_images_binarized = np.array([image.img_to_array(char_image) / 255.0 for char_image in char_images])
        if not char_images_binarized.any():
            return predicted_string, total_confidence
        predictions = self.model.predict(char_images_binarized, verbose=0)
        prediction_classes = np.argmax(predictions, axis=1)
        for prediction, pred_class in zip(predictions, prediction_classes):
            confidence = prediction[pred_class]
            if confidence < self.prediction_threshold:
                predicted_string += "?"
                continue
            predicted_string += self.label_map[int(pred_class)]
            total_confidence += confidence
        avg_conf = total_confidence / len(prediction_classes) if prediction_classes.any() else 0
        return predicted_string, avg_conf