import cv2
import sys
from segmentation import get_character_images
from detection import Detection

# Example usage
if __name__ == "__main__":
    model_yolo_path = "runs/detect/train10/weights/best.pt"
    model_cnn_path = "cnn_model/best.keras"
    debug_dir = "debug/number_plates"
    output_dir = "results"
    if len(sys.argv) != 3:
        print("Error: [--full-image, --full-video, --segmentation] [file_path]")

    elif sys.argv[1] == "--full-image":
        image = cv2.imread(sys.argv[2])
        detection = Detection(
            model_yolo_path=model_yolo_path,
            model_cnn_path=model_cnn_path,
            debug=True,
            debug_dir=debug_dir,
            output_dir=output_dir
        )
        detection.detect_image(image)

    elif sys.argv[1] == "--full-video":
        detection = Detection(
            model_yolo_path=model_yolo_path,
            model_cnn_path=model_cnn_path,
            debug=True,
            debug_dir=debug_dir,
            output_dir=output_dir
        )
        capture = cv2.VideoCapture(sys.argv[2])
        detection.detect_video(capture)
    
    elif sys.argv[1] == "--segmentation":
        plate_image = cv2.imread(sys.argv[2])
        get_character_images(plate_image, debug=True)
        print("Segmented characters save in debug/segmentation/ directory")