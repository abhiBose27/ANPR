import cv2
from prediction import get_plate_string

if __name__ == "__main__":
    image_path = "test_images/test_plate.png"
    plate_image = cv2.imread(image_path)
    plate_string = get_plate_string(plate_image)
    print("🔤 Recognized Plate:", plate_string)