import os

def delete_files_in_directory(directories):
    for directory in directories:
        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                # Only delete files, not directories
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            print("All files deleted successfully.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    delete_files_in_directory(["debug/number_plates", "debug/preprocessing/blurred", "debug/preprocessing/contours", "debug/preprocessing/dilation", 
                               "debug/preprocessing/eroded", "debug/preprocessing/gray", "debug/preprocessing/histo_equ", "debug/preprocessing/thresh", "debug/segmentation"])