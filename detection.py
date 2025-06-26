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
                detected_boxes.append(box)
        return detected_boxes
    
    def get_plate_images(self, plate_boxes, image):
        plate_images = []
        for i, box in enumerate(plate_boxes):
            x1, y1, x2, y2 = map(int, box)
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

def get_highest_area_box(boxes):
    highest_area = 0
    highest_area_box = None
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > highest_area:
            highest_area = box_area
            highest_area_box = box
    return highest_area_box

