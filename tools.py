import numpy as np
import cv2

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def get_label_map():
    class_names = {}
    for i in range(36):
        if i < 10:
            class_names[i] = chr(ord('0') + i)
            continue
        class_names[i] = chr(ord('A') + (i - 10))
    return class_names

def draw_prediction_canvas(image, plate_box, plate_string):
    x1, y1, x2, y2 = plate_box
    h, w = image.shape[:2]

    # Step 1: Dynamic font scale based on image width
    base_font_scale = w / 1000  # adjust this factor to control relative size
    font_scale = max(0.5, min(2, base_font_scale))  # clamp font size

    (text_width, text_height), _ = cv2.getTextSize(
        plate_string,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        thickness=2
    )

    # Step 2: Define minimum canvas size based on image
    min_width = int(w * 0.15)     # e.g., 15% of image width
    min_height = int(h * 0.05)    # e.g., 5% of image height

    # Step 3: Final canvas size
    canvas_width = max(text_width + 10, min_width)
    canvas_height = max(text_height + 20, min_height)
    margin = 10

    # Try to place canvas to the right
    cx1 = x2 + margin
    cy1 = y1
    cx2 = cx1 + canvas_width
    cy2 = cy1 + canvas_height

    if cx2 > w:
        cx2 = x1 - margin
        cx1 = cx2 - canvas_width

    # If it still doesn't fit, try above
    if cx1 < 0:
        cx1 = x1
        cx2 = x1 + canvas_width
        cy2 = y1 - margin
        cy1 = cy2 - canvas_height

    # If all else fails, draw below the plate
    if cy1 < 0:
        cx1 = x1
        cx2 = x1 + canvas_width
        cy1 = y2 + margin
        cy2 = cy1 + canvas_height

    # Draw filled black rectangle
    cv2.rectangle(image, (cx1, cy1), (cx2, cy2), (0, 0, 0), -1)

    text_x = cx1 + (canvas_width - text_width) // 2
    text_y = cy1 + (canvas_height + text_height) // 2 - 5  # vertically centered baseline

    # Draw text inside it
    cv2.putText(
        image,
        plate_string,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2
    )

    # Also draw bounding box on the plate
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)