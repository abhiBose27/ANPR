import cv2
from collections import defaultdict

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

def get_detected_boxes(model, image, conf_threshold=0.5):
    #model = YOLO(model_path)
    result = model(image)[0]
    class_index_to_boxes = defaultdict(list)
    detected_boxes = []
    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        class_index = int(cls.item())
        if conf.item() > conf_threshold:
            class_index_to_boxes[class_index].append(box.cpu().numpy())
    for _, boxes in class_index_to_boxes.items():
        detected_boxes.append(get_highest_area_box(boxes))
    return detected_boxes

def crop(image, box):
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]

def get_plate_images(plate_boxes, image, output_dir="debug/number_plates/", debug=False):
    #boxes = get_detected_boxes(model, image_path)
    plate_images = []
    for i, box in enumerate(plate_boxes):
        plate_image = crop(image, box)
        if debug:
            cv2.imwrite(f"{output_dir}/0_original.jpg", plate_image)
        plate_images.append(plate_image)
    return plate_images

"""
def process_plate_image(image, debug, output_dir):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 41, 15)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    mask = np.zeros_like(thresh)
    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] > 30:  # Ignore noise
            mask[labels == i] = 255
    #edges = auto_canny(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        debug_img = image.copy()
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
        debug_img = image.copy()
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imwrite(f"{output_dir}/debug_character_boxes.jpg", debug_img)

    char_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]

    if not char_boxes:
        print("⚠️ No valid character contours found.")
        return image
    
    x_coords = [x for x, _, w, _ in char_boxes] + [x+w for x, _, w, _ in char_boxes]
    y_coords = [y for _, y, _, h in char_boxes] + [y+h for _, y, _, h in char_boxes]

    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)
    cropped = image[y1:y2, x1:x2] 

    if debug:
        cv2.imwrite(f"{output_dir}/0_original.jpg", image)
        cv2.imwrite(f"{output_dir}/1_gray.jpg", gray)
        cv2.imwrite(f"{output_dir}/2_blurred.jpg", blurred)
        cv2.imwrite(f"{output_dir}/3_thresh.jpg", thresh)
        cv2.imwrite(f"{output_dir}/4_mask.jpg", mask)
        cv2.imwrite(f"{output_dir}/5_cropped.jpg", cropped)
        #cv2.imwrite(f"{output_dir}/6_edge_detection.jpg", edges)

    # Crop
    return cropped

    if debug:
        path = os.path.join(output_dir, "step0_gray.jpg")
        cv2.imwrite(path, gray)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 12)
    if debug:
        path = os.path.join(output_dir, "step1_thresh.jpg")
        cv2.imwrite(path, thresh)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)

    #edges = auto_canny(cleaned)
    mask = np.zeros_like(cleaned)
    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] > 30:  # Ignore noise
            mask[labels == i] = 255
    
    # Get tight bounds with 3px safety margin
    x,y,w,h = cv2.boundingRect(mask)
    x1 = max(0, x-3)
    y1 = max(0, y-3)
    x2 = min(image.shape[1], x+w+3)
    y2 = min(image.shape[0], y+h+3)
    
    # Final crop
    cropped = image[y1:y2, x1:x2]
    
    if debug:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f"{output_dir}/1_original.jpg", image)
        cv2.imwrite(f"{output_dir}/2_threshold.jpg", thresh)
        cv2.imwrite(f"{output_dir}/3_mask.jpg", mask)
        cv2.imwrite(f"{output_dir}/4_cropped.jpg", cropped)
    
    return cropped

    if debug:
        path = os.path.join(output_dir, "step2_edges.jpg")
        cv2.imwrite(path, edges)

    # Find largest bounding box around text
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    if not char_boxes:
        print("⚠️ No valid character contours found.")
        return image
    
    x_coords = [x for x, _, w, _ in char_boxes] + [x+w for x, _, w, _ in char_boxes]
    y_coords = [y for _, y, _, h in char_boxes] + [y+h for _, y, _, h in char_boxes]

    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)

    # Crop
    return image[y1:y2, x1:x2]"""
