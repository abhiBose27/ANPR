import cv2
import sys
from segmentation import get_character_images
from detection import Detection
    

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Need args")
    if sys.argv[1] == "--detection":
        detection = Detection(
            model_yolo_path="runs/detect/train5/weights/best.onnx",
            model_cnn_path="cnn_model/best.keras",
            debug=True,
            debug_dir="debug/number_plates/"
        )
        image = cv2.imread(sys.argv[2])
        plate_boxes = detection.get_detected_boxes(image)
        detection.get_plate_images(plate_boxes, image)
        print("Number plate Images saved in debug/number_plates/ directory")
    
    elif sys.argv[1] == "--full-image":
        image = cv2.imread(sys.argv[2])
        detection = Detection(
            model_yolo_path="runs/detect/train5/weights/best.onnx",
            model_cnn_path="cnn_model/best.keras",
            debug=True,
            debug_dir="debug/number_plates/"
        )
        detection.detect(image)

    elif sys.argv[1] == "--full-video":
        detection = Detection(
            model_yolo_path="runs/detect/train5/weights/best.onnx",
            model_cnn_path="cnn_model/best.keras",
            debug=False,
            debug_dir="debug/number_plates/"
        )
        capture = cv2.VideoCapture(sys.argv[2])
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            detection.detect(frame)
            frame = cv2.resize(frame, (1280, 780))
            cv2.imshow("Number Plate recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
    
    elif sys.argv[1] == "--segmentation":
        plate_image = cv2.imread(sys.argv[2])
        get_character_images(plate_image, debug=True)
        print("Segmented characters save in debug/segmentation/ directory")
    
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
