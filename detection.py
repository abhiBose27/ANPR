import cv2
import datetime
import numpy as np
from tools import draw_prediction_canvas
from prediction import Prediction
from ultralytics import YOLO
from sort.sort import Sort

class Detection:
    def __init__(self, model_yolo_path, model_cnn_path, debug, debug_dir):
        self.debug = debug
        self.debug_dir = debug_dir
        self.seen_plate_ids = {}
        self.tracker = Sort()
        self.model = YOLO(model_yolo_path)
        self.prediction = Prediction(model_cnn_path, debug) 

    def get_detected_boxes(self, image, conf_threshold=0.7):
        result = self.model(image, imgsz=320, conf=0.6)[0]
        detected_boxes = []
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = box
            #if conf > conf_threshold:
            print(conf)
            detected_boxes.append([x1, y1, x2, y2, conf])
        return detected_boxes
    
    def detect_image(self, image):
        plate_boxes = self.get_detected_boxes(image)
        for i, plate_box in enumerate(plate_boxes):
            x1, y1, x2, y2, conf = map(int, plate_box)
            plate_image = image[y1 : y2, x1 : x2]
            result = self.prediction.get_plate_string(plate_image)
            plate_string, confidence = result
            print(f"Plate: {plate_string} | Confidence: {confidence:.2f}")
            draw_prediction_canvas(image, (x1, y1, x2, y2), plate_string)
            if self.debug:
                ts = datetime.datetime.now().timestamp() * 1000000
                cv2.imwrite(f"{self.debug_dir}/{ts}_{plate_string}_plate.jpg", plate_image)
            print("🔤 Recognized Plate:", plate_string)

    def detect_video(self, frame):
        plate_boxes = self.get_detected_boxes(frame)
        plate_boxes_np = np.array(plate_boxes, dtype=np.float32)

        if plate_boxes_np.size == 0:
            plate_boxes_np = np.empty((0, 5), dtype=np.float32)
        elif plate_boxes_np.ndim == 1:
            plate_boxes_np = np.expand_dims(plate_boxes_np, axis=0)
        tracker_boxes = self.tracker.update(plate_boxes_np)

        for tracker_box in tracker_boxes:
            x1, y1, x2, y2, plate_id = map(int, tracker_box)
            plate_image = frame[y1:y2, x1:x2]
            if plate_image.size == 0:
                continue
            if plate_id not in self.seen_plate_ids:
                result = self.prediction.get_plate_string(plate_image)
                plate_string, confidence = result
                if confidence >= 0.8:
                    self.seen_plate_ids[plate_id] = {
                        "confidence_lvl": "confident",
                        "licence": plate_string,
                        "plate_image": plate_image
                    }
                if 0.4 <= confidence < 0.8:
                    self.seen_plate_ids[plate_id] = {
                        "confidence_lvl": "tentative",
                        "licence": plate_string,
                        "plate_image": plate_image
                    }
            else:
                plate_info = self.seen_plate_ids[plate_id]
                if plate_info["confidence_lvl"] == "tentative":
                    result = self.prediction.get_plate_string(plate_image)
                    plate_string, confidence = result
                    if confidence >= 0.8:
                        self.seen_plate_ids[plate_id] = {
                            "confidence_lvl": "confident",
                            "licence": plate_string,
                            "plate_image": plate_image
                        }
                    
            plate_info = self.seen_plate_ids.get(plate_id, {
                "licence": "reading...", 
                "plate_image": None
            })
            draw_prediction_canvas(frame, (x1, y1, x2, y2), plate_info["licence"])
    
    def log_seen_plate_info(self):
        if not self.debug:
            return
        for plate_id in self.seen_plate_ids:
            ts = datetime.datetime.now().timestamp() * 1000000
            cv2.imwrite(f"{self.debug_dir}/{ts}_{self.seen_plate_ids[plate_id]["licence"]}_plate.jpg", 
                        self.seen_plate_ids[plate_id]["plate_image"])
