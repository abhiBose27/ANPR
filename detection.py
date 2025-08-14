import os
import cv2
import datetime
import numpy as np
from tools import draw_prediction_canvas
from segmentation import Segmentation
from preprocessing import Preprocessing
from prediction import OCREngine
from ultralytics import YOLO
from sort.sort import Sort


class PlateManager:
    def __init__(self, debug, debug_dir):
        self.debug = debug
        self.debug_dir = debug_dir
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
        self.tracker = Sort()
        self.seen_plate_ids = {}
    
    def get_updated_tracked_plate_boxes(self, plate_boxes):
        plate_boxes_np = np.array(plate_boxes, dtype=np.float32)
        if plate_boxes_np.size == 0:
            plate_boxes_np = np.empty((0, 5), dtype=np.float32)
        elif plate_boxes_np.ndim == 1:
            plate_boxes_np = np.expand_dims(plate_boxes_np, axis=0)
        return self.tracker.update(plate_boxes_np)
    
    def log_seen_plate_info(self):
        if not self.debug:
            return
        for plate_id in self.seen_plate_ids:
            ts = datetime.datetime.now().timestamp() * 1000000
            cv2.imwrite(f"{self.debug_dir}/{ts}_{self.seen_plate_ids[plate_id]["licence"]}_plate.jpg", 
                        self.seen_plate_ids[plate_id]["plate_image"])
    
    def is_plate_ocr_required(self, plate_id):
        return plate_id not in self.seen_plate_ids or '?' in self.seen_plate_ids[plate_id]["licence"]
    
    def update_seen_plate_ids(self, plate_id, plate_info):
        self.seen_plate_ids[plate_id] = plate_info
    
    def get_plate_info(self, plate_id):
        return self.seen_plate_ids.get(plate_id, {
            "licence": "reading...", 
            "plate_image": None
        })

class Detection:
    def __init__(self, model_yolo_path, model_cnn_path, debug, output, debug_dir, output_dir):
        self.debug = debug
        self.output = output
        self.debug_dir = debug_dir
        self.output_dir = output_dir
        self.model = YOLO(model_yolo_path)
        self.prediction = OCREngine(model_cnn_path)
        self.preprocessing = Preprocessing(self.debug)
        self.segmentation = Segmentation(self.debug)
        self.plate_manager = PlateManager(self.debug, self.debug_dir)
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
        if self.output:
            os.makedirs(self.output_dir, exist_ok=True)

    def get_plate_boxes(self, image):
        result = self.model(image, imgsz=640, conf=0.7)[0]
        detected_boxes = []
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, _ = box
            print(f"YOLO Confidence: {conf:.2f}")
            detected_boxes.append([x1, y1, x2, y2, conf])
        return detected_boxes
    
    def detect_image(self, image):
        plate_boxes = self.get_plate_boxes(image)
        for i, plate_box in enumerate(plate_boxes):
            x1, y1, x2, y2, _ = map(int, plate_box)
            plate_image = image[y1 : y2, x1 : x2]

            preprocessed_image = self.preprocessing.get_preprocessed_image(plate_image)
            bounding_boxes = self.preprocessing.get_bounding_boxes(preprocessed_image, plate_image.copy())
            char_images = self.segmentation.get_character_images(preprocessed_image, bounding_boxes)
            plate_string, confidence = self.prediction.run_batch_ocr(char_images)

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
        self.plate_manager.log_seen_plate_info()

    def process_frame(self, frame):
        plate_boxes = self.get_plate_boxes(frame)
        tracked_plate_boxes = self.plate_manager.get_updated_tracked_plate_boxes(plate_boxes)
        for tracked_box in tracked_plate_boxes:
            x1, y1, x2, y2, plate_id = map(int, tracked_box)
            plate_image = frame[y1:y2, x1:x2]
            if plate_image.size == 0:
                continue
            
            if self.plate_manager.is_plate_ocr_required(plate_id):
                preprocessed_image = self.preprocessing.get_preprocessed_image(plate_image)
                bounding_boxes = self.preprocessing.get_bounding_boxes(preprocessed_image, plate_image.copy())
                char_images = self.segmentation.get_character_images(preprocessed_image, bounding_boxes)
                plate_string, confidence = self.prediction.run_batch_ocr(char_images)

                if len(plate_string) <= 8 and len(plate_string) >= 6 and confidence >= 0.8:
                    self.plate_manager.update_seen_plate_ids(plate_id, {
                        "confidence_lvl": confidence,
                        "licence": plate_string,
                        "plate_image": plate_image
                    })
            plate_info = self.plate_manager.get_plate_info(plate_id)
            draw_prediction_canvas(frame, (x1, y1, x2, y2), plate_info["licence"])
