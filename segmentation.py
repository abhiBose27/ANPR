import cv2
import os
import numpy as np
from tools import draw_contours


def get_preprocessed_image(plate_image, output_dir, debug):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)

    # Preprocessing: adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 41, 12)
    
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)

    if debug:
        cv2.imwrite(f"{output_dir}/0_gray.jpg", gray)
        cv2.imwrite(f"{output_dir}/1_blurred.jpg", blurred)
        cv2.imwrite(f"{output_dir}/2_thresh.jpg", thresh)
        cv2.imwrite(f"{output_dir}/3_mask.jpg", dilation)
    return dilation

def get_bounding_boxes(preprocessed_image):
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes       = [cv2.boundingRect(c) for c in contours]
    sorted_boxes = sorted(boxes, key=lambda box: box[0])
    bounding_boxes = []
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        roi_area = w * h
        if h / float(w) < 1.2:
            continue
        if roi_area < 100:
            continue
        bounding_boxes.append((x, y, w, h))
    median_height = np.median([h for _, _, _, h in bounding_boxes])
    bounding_boxes = [(x, y, w, h) for x, y, w, h in bounding_boxes if h >= 0.8 * median_height]
    return bounding_boxes

def get_character_images(plate_image, output_dir="debug/segmentation", debug=False):
    preprocessed_image = get_preprocessed_image(plate_image, output_dir, debug)
    sorted_boxes = get_bounding_boxes(preprocessed_image)
    
    char_images = []
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        char = preprocessed_image[y : y + h, x : x + w]
        char = cv2.resize(char, (28, 28), interpolation=cv2.INTER_AREA)
        char_images.append(char)
    if debug:
        draw_contours(plate_image, sorted_boxes, output_dir)
    return char_images
