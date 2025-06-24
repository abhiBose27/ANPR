import cv2
import sys
from ultralytics import YOLO
from segmentation import get_character_images
from detection import get_plate_images, get_detected_boxes
from prediction import get_plate_string
from tools import draw_prediction_canvas

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Need args")
    if sys.argv[1] == "--detection":
        model = YOLO("runs/detect/train/weights/best.pt")
        image = cv2.imread(sys.argv[2])
        plate_boxes = get_detected_boxes(model, sys.argv[2])
        get_plate_images(plate_boxes, sys.argv[2], debug=True)
        print("Number plate Images saved in debug/number_plates/ directory")
    
    elif sys.argv[1] == "--segmentation":
        plate_image = cv2.imread(sys.argv[2])
        get_character_images(plate_image, debug=True)
        print("Segmented characters save in debug/segmentation/ directory")
    
    elif sys.argv[1] == "--full":
        nb_plates = 0
        model = YOLO("runs/detect/train/weights/best.pt")
        image = cv2.imread(sys.argv[2])
        plate_boxes = get_detected_boxes(model, image)
        plate_images = get_plate_images(plate_boxes, image, debug=True)
        for plate_image, plate_box in zip(plate_images, plate_boxes):
            plate_string = get_plate_string(plate_image)
            image_copy = image.copy()
            draw_prediction_canvas(image_copy, plate_box, plate_string)
            cv2.imwrite(f"debug/number_plates/{nb_plates}_plates.jpg", image_copy)
            print("🔤 Recognized Plate:", plate_string)
            nb_plates += 1
    else:
        raise Exception("Unexpected")


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import os
# import sys

# # Load model
# model = tf.keras.models.load_model('cnn_model.h5')

# # Map class indices (from training)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(rescale=1./255)
# generator = datagen.flow_from_directory(
#     'dataset/',
#     target_size=(28, 28),
#     color_mode='grayscale',
#     class_mode='categorical',
#     batch_size=1,
#     shuffle=False
# )
# class_indices = generator.class_indices
# label_map = {v: k for k, v in class_indices.items()}

# def predict_character(img_path):
#     img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     pred = model.predict(img_array)
#     predicted_class = np.argmax(pred)
#     character = label_map[predicted_class]

#     print(f"🔤 Predicted Character: {character}")

# # Example usage
# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print("❌ Usage: python predict_character.py path/to/image.png")
#     else:
#         predict_character(sys.argv[1])
