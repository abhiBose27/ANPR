import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add this at the top:
model = tf.keras.models.load_model('cnn_model.h5')
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory('dataset/', target_size=(28, 28), color_mode='grayscale', class_mode='categorical', batch_size=1)
class_indices = generator.class_indices
label_map = {v: k for k, v in class_indices.items()}

def predict_image(path):
    img = image.load_img(path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return label_map[np.argmax(pred)]




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
