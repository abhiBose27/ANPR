import os
import cv2
import datetime
import numpy as np
from tools import draw_prediction_canvas
from prediction import Prediction
from ultralytics import YOLO
from sort.sort import Sort

class Detection:
    def __init__(self, model_yolo_path, model_cnn_path, debug, output, debug_dir, output_dir):
        self.debug = debug
        self.output = output
        self.debug_dir = debug_dir
        self.output_dir = output_dir
        self.seen_plate_ids = {}
        self.tracker = Sort()
        self.model = YOLO(model_yolo_path)
        self.prediction = Prediction(model_cnn_path, debug)
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
        if self.output:
            os.makedirs(self.output_dir, exist_ok=True)

    def get_detected_boxes(self, image):
        result = self.model(image, imgsz=320, conf=0.6)[0]
        detected_boxes = []
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, _ = box
            print(f"YOLO Confidence: {conf:.2f}")
            detected_boxes.append([x1, y1, x2, y2, conf])
        return detected_boxes
    
    def detect_image(self, image):
        plate_boxes = self.get_detected_boxes(image)
        for i, plate_box in enumerate(plate_boxes):
            x1, y1, x2, y2, _ = map(int, plate_box)
            plate_image = image[y1 : y2, x1 : x2]
            plate_string, confidence = self.prediction.get_plate_string(plate_image)
            print(f"Plate: {plate_string} | Confidence: {confidence:.2f}")
            draw_prediction_canvas(image, (x1, y1, x2, y2), plate_string)
            if self.debug:
                ts = datetime.datetime.now().timestamp() * 1000000
                cv2.imwrite(f"{self.debug_dir}/{ts}_{plate_string}_plate.jpg", plate_image)
        if self.output:
            ts = datetime.datetime.now().timestamp() * 1000000
            cv2.imwrite(f"{self.output_dir}/{ts}_{plate_string}_plate.jpg", image)

    def detect_video(self, capture):
        if self.output:
            fps = int(capture.get(cv2.CAP_PROP_FPS) / 2)
            width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mpv4")
            ts = datetime.datetime.now().timestamp() * 1000000
            out = cv2.VideoWriter(f"{self.output_dir}/{ts}_video.mp4", fourcc, fps, (width, height))

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            self.process_frame(frame)
            cv2.imshow("Number Plate recognition", cv2.resize(frame, (1280, 780)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if self.output:
                out.write(frame)
        capture.release()
        if self.output:
            out.release()
        cv2.destroyAllWindows()
        self.log_seen_plate_info()
    
    def process_frame(self, frame):
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
                plate_string, confidence = self.prediction.get_plate_string(plate_image)
                if confidence >= 0.8 and len(plate_string) >= 6:
                    self.seen_plate_ids[plate_id] = {
                        "confidence_lvl": "confident",
                        "licence": plate_string,
                        "plate_image": plate_image
                    }
                if 0.4 <= confidence < 0.8 and len(plate_string) >= 6:
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
                    if confidence >= 0.8 and len(plate_string) >= 6:
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
