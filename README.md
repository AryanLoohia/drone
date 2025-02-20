# Object Detection and Hazard Analysis

This project uses the YOLO object detection model to detect objects in an image and analyze the hazard level of each object. The script utilizes the Groq AI API to generate hazard analysis for each detected object. The detected objects are displayed with bounding boxes, and the hazard analysis is printed and saved to a text file.

## Requirements

- Python 3.9
- OpenCV
- NumPy
- Groq AI API

## Installation

1. Clone the repository:
    
    git clone https://github.com/AryanLoohia/drone.git
    cd <repository_directory>
    

2. Install the required Python packages:
    
    pip install opencv-python numpy groq
    

## Usage

To run the object detection and hazard analysis script, use the following command:

python obj_DETECTION.py -i <path_to_input_image> -c <path_to_yolo_config_file> -w <path_to_yolo_weights_file> -cl <path_to_classes_file>

python obj_DETECTION.py -i img1.jpg -c yolov3.cfg -w yolov3.weights -cl yolov3.txt
