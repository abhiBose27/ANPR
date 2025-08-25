# ANPR (Automatic Number Plate Recognition)

As the name suggests, real-time number plate recognition system using Computer Vision and OCR techniques.

## Pipeline

The pipeline is as follows:-
 1) Detecting number plates using YOLOv8 deep learning model which is trained on over 5000 training images.
 2) Pre-processing of detected and cropped number plates with computer vision techniques (grayscale, histogram equalization, gaussian blur, adaptive thresholding, erosion, dilation).
 3) Segmentation of characters using contour filtration and running a batch OCR on our own OCR Engine(a multi-layered CNN model).
 4) Prediction and drawing on images/frames.

## How to run the whole pipeline?

```
python main.py [--full-image, --full-video, --segmentation] [file_path] Optional: [--debug --output]
```
