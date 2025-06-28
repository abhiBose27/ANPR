import cv2
import datetime
import numpy as np
from tools import draw_contours


def _get_preprocessed_image(plate_image, debug_dir, debug):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)

    # Preprocessing: adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 41, 12)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(eroded, kernel, iterations=3)
    ts = datetime.datetime.now().timestamp() * 1000000
    if debug:
        cv2.imwrite(f"{debug_dir}/gray/{ts}_gray.jpg", gray)
        cv2.imwrite(f"{debug_dir}/blurred/{ts}_blurred.jpg", blurred)
        cv2.imwrite(f"{debug_dir}/thresh/{ts}_thresh.jpg", thresh)
        cv2.imwrite(f"{debug_dir}/eroded/{ts}_eroded.jpg", eroded)
        cv2.imwrite(f"{debug_dir}/dilation/{ts}_dilation.jpg", dilation)
    return dilation

def _get_bounding_boxes(preprocessed_image):
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes       = [cv2.boundingRect(c) for c in contours]
    sorted_boxes = sorted(boxes, key=lambda box: box[0])
    bounding_boxes = []
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        roi_area = w * h
        aspect_ratio = w / float(h)
        roi = preprocessed_image[y : y + h, x : x + w]
        black_pixels = (w * h) - cv2.countNonZero(roi)  # white = nonzero
        black_ratio = black_pixels / (w * h)
        if w > h or not (0.2 < aspect_ratio < 0.8) or roi_area < 100:
            continue
        if not (0.25 < black_ratio < 0.8):
            continue
        bounding_boxes.append((x, y, w, h))
    median_height = np.median([h for _, _, _, h in bounding_boxes])
    bounding_boxes = [(x, y, w, h) for x, y, w, h in bounding_boxes if h >= 0.8 * median_height]
    return bounding_boxes

def get_character_images(plate_image, debug, debug_dir="debug/segmentation"):
    preprocessed_image = _get_preprocessed_image(plate_image, debug_dir, debug)
    sorted_boxes = _get_bounding_boxes(preprocessed_image)
    char_images = []
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        char = preprocessed_image[y : y + h, x : x + w]
        char = cv2.resize(char, (28, 28), interpolation=cv2.INTER_AREA)
        char_images.append(char)
    if debug:
        draw_contours(plate_image, sorted_boxes, debug_dir)
    return char_images
