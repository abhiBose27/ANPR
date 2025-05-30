from segment_plate import segment_characters
from predict_character import predict_image

def recognize_plate_string(plate_image_path):
    char_paths = segment_characters(plate_image_path)
    predicted_string = ''.join([predict_image(p) for p in char_paths])
    return predicted_string

# Example usage
if __name__ == "__main__":
    plate_img = "test_plate.png"  # Replace with your image
    result = recognize_plate_string(plate_img)
    print("🔤 Recognized Plate:", result)
