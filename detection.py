import cv2
from tools import draw_prediction_canvas
from prediction import Prediction
from ultralytics import YOLO

class Detection:
    def __init__(self, model_yolo_path, model_cnn_path, debug, debug_dir):
        self.debug = debug
        self.debug_dir = debug_dir
        self.model = YOLO(model_yolo_path)
        self.prediction = Prediction(model_cnn_path, debug)

    def get_detected_boxes(self, image, conf_threshold=0.5):
        result = self.model(image, imgsz=320)[0]
        detected_boxes = []
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if conf.item() > conf_threshold:
                x1, y1, x2, y2 = map(int, box)
                detected_boxes.append((x1, y1, x2, y2))
        return detected_boxes
    
    def get_plate_images(self, plate_boxes, image):
        plate_images = []
        for i, (x1, y1, x2, y2) in enumerate(plate_boxes):
            plate_image = image[y1:y2, x1:x2]
            if self.debug:
                cv2.imwrite(f"{self.debug_dir}/{i}_original.jpg", plate_image)
            plate_images.append(plate_image)
        return plate_images
    
    def detect(self, image):
        nb_plates = 0
        plate_boxes = self.get_detected_boxes(image)
        plate_images = self.get_plate_images(plate_boxes, image)
        for plate_image, plate_box in zip(plate_images, plate_boxes):
            plate_string = self.prediction.get_plate_string(plate_image)
            if plate_string == "":
                continue
            draw_prediction_canvas(image, plate_box, plate_string)
            if self.debug:
                cv2.imwrite(f"{self.debug_dir}/{nb_plates}_plates.jpg", image)
            nb_plates += 1
            print("🔤 Recognized Plate:", plate_string)


