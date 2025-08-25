import sys

# Example usage
if __name__ == "__main__":
    model_yolo_path = "runs/detect/train4/weights/best.pt"
    model_cnn_path = "cnn_model/best.keras"
    debug_dir = "debug/number_plates"
    output_dir = "results"
    debug = False
    output = False
    image_detection = None
    video_detection = None

    for i in range(len(sys.argv)):
        if i == 0:
            continue
        if sys.argv[i] == "--debug":
            debug = True
        if sys.argv[i] == "--output":
            output = True
        if sys.argv[i] == "--full-image" and i + 1 < len(sys.argv):
            image_detection = sys.argv[i + 1]
        if sys.argv[i] == "--full-video" and i + 1 < len(sys.argv):
            video_detection = sys.argv[i + 1]

    if not video_detection or not image_detection:
        print("Error: [--full-image, --full-video, --segmentation] [file_path] Optional: [--debug --output]")
        exit(1)

    import cv2
    from detection import Detection

    detection = Detection(
        model_yolo_path=model_yolo_path,
        model_cnn_path=model_cnn_path,
        debug=debug,
        output=output,
        debug_dir=debug_dir,
        output_dir=output_dir
    )
    if video_detection:
        capture = cv2.VideoCapture(video_detection)
        detection.detect_video(capture)
    elif image_detection:
        image = cv2.imread(image_detection)
        detection.detect_image(image)

