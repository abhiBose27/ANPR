import cv2
from prediction import Prediction

if __name__ == "__main__":
    image_path = "test_images/test_plate.png"
    plate_image = cv2.imread(image_path)
    prediction = Prediction("cnn_model/best.keras", debug=True)
    plate_string = prediction.get_plate_string(plate_image)
    print("🔤 Recognized Plate:", plate_string)