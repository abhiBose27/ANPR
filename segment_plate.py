import cv2
import os

def segment_characters(image_path, output_dir="segmented_chars"):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    sorted_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    char_images = []
    for i, (x, y, w, h) in enumerate(sorted_boxes):
        if w > 5 and h > 10:  # Filter noise
            char = gray[y:y+h, x:x+w]
            char = cv2.resize(char, (28, 28))
            char_path = os.path.join(output_dir, f"char_{i}.png")
            cv2.imwrite(char_path, char)
            char_images.append(char_path)

    return char_images
