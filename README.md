# ANPR (Automatic Number Plate Recognition)

As the name suggests, real-time number plate recognition system using Computer Vision and OCR techniques.

## Pipeline

The pipeline is as follows:-
✅ Detecting number plates using YOLOv8 deep learning model which is trained on over 5000 training images.
✅ Pre-processing of detected and cropped number plates with computer vision techniques (grayscale, histogram equalization, gaussian blur, adaptive thresholding, erosion, dilation).
✅ Segmentation of characters using contour filtration and running a batch OCR on our own OCR Engine(a multi-layered CNN model).
✅ Prediction and drawing on images/frames.

## How to run the whole pipeline?

```
python main.py [--full-image, --full-video, --segmentation] [file_path] Optional: [--debug --output]
```
