import cv2
import os
import datetime
import numpy as np


class Preprocessing:
    def __init__(self, debug, debug_dir="debug/preprocessing"):
        self.debug = debug
        self.debug_dir = debug_dir
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

    def get_preprocessed_image(self, plate_image):
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)

        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 41, 12)
    
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        eroded = cv2.erode(thresh, kernel, iterations=1)
        dilation = cv2.dilate(eroded, kernel, iterations=2)
        ts = datetime.datetime.now().timestamp() * 1000000
        if self.debug:
            os.makedirs(f"{self.debug_dir}/gray", exist_ok=True)
            os.makedirs(f"{self.debug_dir}/blurred", exist_ok=True)
            os.makedirs(f"{self.debug_dir}/thresh", exist_ok=True)
            os.makedirs(f"{self.debug_dir}/eroded", exist_ok=True)
            os.makedirs(f"{self.debug_dir}/dilation", exist_ok=True)

            cv2.imwrite(f"{self.debug_dir}/gray/{ts}_gray.jpg", gray)
            cv2.imwrite(f"{self.debug_dir}/blurred/{ts}_blurred.jpg", blurred)
            cv2.imwrite(f"{self.debug_dir}/thresh/{ts}_thresh.jpg", thresh)
            cv2.imwrite(f"{self.debug_dir}/eroded/{ts}_eroded.jpg", eroded)
            cv2.imwrite(f"{self.debug_dir}/dilation/{ts}_dilation.jpg", dilation)
        return dilation
    
    def get_bounding_boxes(self, preprocessed_image, plate_image):
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxes       = [cv2.boundingRect(c) for c in contours]
        sorted_boxes = sorted(boxes, key=lambda box: box[0])
        bounding_boxes = []

        for i, (x, y, w, h) in enumerate(sorted_boxes):
            roi_area = w * h
            aspect_ratio = w / float(h)
            roi = preprocessed_image[y : y + h, x : x + w]
            black_pixels = (w * h) - cv2.countNonZero(roi)
            black_ratio = black_pixels / (w * h)
            if w > h or not (0.2 < aspect_ratio < 0.8) or roi_area < 500 or not (0.25 < black_ratio < 0.8):
                continue
            bounding_boxes.append((x, y, w, h))
        median_height = np.median([h for _, _, _, h in bounding_boxes])
        bounding_boxes = [(x, y, w, h) for x, y, w, h in bounding_boxes if  h / median_height >= 0.8]
        if self.debug:
            os.makedirs(f"{self.debug_dir}/contours", exist_ok=True)
            for x, y, w, h in bounding_boxes:
                cv2.rectangle(plate_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            ts = datetime.datetime.now().timestamp() * 1000000
            cv2.imwrite(f"{self.debug_dir}/contours/{ts}_contours.jpg", plate_image)
        return bounding_boxes
