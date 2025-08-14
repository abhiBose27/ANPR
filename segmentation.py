import cv2
import os
import datetime

class Segmentation:
    def __init__(self, debug, debug_dir="debug/segmentation"):
        self.debug = debug
        self.debug_dir = debug_dir
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
    
    def get_character_images(self, preprocessed_image, bounding_boxes):
        char_images = []
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            char = preprocessed_image[y : y + h, x : x + w]
            char = cv2.resize(char, (28, 28), interpolation=cv2.INTER_AREA)
            char_images.append(char)
        
        # Save character images
        if self.debug:
            ts = datetime.datetime.now().timestamp() * 1000000
            for i, char_image in enumerate(char_images):
                cv2.imwrite(f"{self.debug_dir}/{ts}_{i}_char.jpg", char_image)
        return char_images

