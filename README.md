# Lane and Traffic Sign Detection Using CNN

## Overview
This project aims to develop a smart self-driving car system using deep learning models for traffic sign detection and steering angle prediction. The system utilizes YOLO for object detection and a custom convolutional neural network for steering control.

## Project Structure
```
.
|-- dataset/
|   |-- shape_predictor_68_face_landmarks.dat
|
|-- model_data/
|   |-- coco_classes.txt
|   |-- tiny_yolo_anchors.txt
|   |-- yolo_anchors.txt
|   |-- yolo_weights.h5
|
|-- models/
|   |-- best.pt       # YOLO model for traffic sign detection
|   |-- Stearing.h5   # Steering angle prediction model
|
|-- Notebook/
|   |-- SelfDriving.ipynb
|
|-- yolo_backend/
|   |-- model.py
|   |-- utils.py
|
|-- scripts/
|   |-- Algorithm.py
|   |-- Distance_Estimator.py
|   |-- Home.py
|   |-- startDrowsy.py
|   |-- sendGmail.py
|   |-- face_detector.py
|   |-- Constant.py
|
|-- README.md
```

## Requirements
- Python 3.7+
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- SciPy
- Pickle
- Scikit-learn
- PIL
- Ultralyics YOLO
- Google Colab (for training/testing)

## Installation
1. Clone the repository:
   ```bash
   git clone  https://github.com/Arshad-khan05/Arshad-khan05-Lane_and_Traffic_Sign_Detection_Using_CNN.git
   ```


## Dataset
- **Traffic Sign Detection**: The YOLO model `best.pt` was trained using the Roboflow dataset.
- **Steering Angle Prediction**: The `Stearing.h5` model was trained on a Kaggle dataset, containing images and steering angles stored in `data.txt`.

## Running the Project
### Running on Images
```python
from Algorithm import detect_image

detect_image('path/to/image.jpg', yolo_detector)
```

### Running on Videos
```python
from Algorithm import detect_video

detect_video('path/to/video.mp4', yolo_detector)
```

### Face Detection
```python
python face_detector.py
```

## Training
To train the steering model:
1. Prepare the dataset by running:
   ```python
   python preprocess_data.py
   ```
2. Train the model:
   ```python
   python train_model.py
   ```

## Features
- **Traffic Sign Detection**: Detects traffic signs like speed limits and signals.
- **Lane Detection**: Identifies lane markings on the road.
- **Steering Control**: Predicts appropriate steering angles based on input images.
- **Drowsiness Detection**: Alerts when the driver appears drowsy.
- **Distance Estimation**: Estimates the distance of detected objects.

## Model Details
- **YOLO (best.pt)**: Trained to recognize traffic-related objects such as:
  - Red light
  - Green light
  - Speed limit signs
- **Steering Prediction Model (Stearing.h5)**:
  - Custom CNN architecture trained on preprocessed steering data

## Results
- Accuracy achieved for traffic sign detection: ~90%
- Steering angle prediction MSE: ~0.02


