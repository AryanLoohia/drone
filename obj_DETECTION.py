# Description: This script uses the YOLO object detection model to detect objects in an image and analyze the hazard level of each object. 
# The script uses the Groq AI API to generate hazard analysis for each object detected in the image. The script displays the image with 
# bounding boxes around detected objects and prints the hazard analysis for each object. The script also writes the hazard analysis to a 
# text file named 'hazard_analysis.txt'.

# Import necessary libraries
import cv2
import argparse
import numpy as np
import os
from groq import Groq

# Define the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

# Define output file
output_file = "hazard_analysis.txt"


with open(output_file, "w") as f:
    f.write("New Image Detected...\n\n")

# YOLO class mappings (COCO dataset)
yolo_classes = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 4: "aeroplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "sofa", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "TV monitor", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

# Initialize Groq AI client
os.environ["GROQ_API_KEY"] = "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"
groq_api_key = os.getenv("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")
client = Groq(api_key=groq_api_key)

# Define class categories
non_hazardous = {11,14,24,25,26,27,28,29,30,31,32,33,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,63,64,65,66,67,70,71,73,74,75,76,77,78,79}
low_hazard = {1, 3,10, 12,13,15,16,18,19,56,57,58,62,68,69,72}
medium_hazard = {2,9,17,22,59,60,61}
high_hazard = {0, 4, 5,6,7,8,20,21,23}
movable_objects = {0,1,2,3,4,5,6,7,8,14,15,16,17,18,19,20,21,22,23,33}

# Function to get YOLO output layers
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Function to determine hazard category
def get_hazard_category(class_id):
    if class_id in non_hazardous:
        return "Non-Hazardous", (0, 255, 0)  # Green
    elif class_id in low_hazard:
        return "Low Hazard", (0, 170, 85)
    elif class_id in medium_hazard:
        return "Medium Hazard", (0, 85, 170)
    elif class_id in high_hazard:
        return "High Hazard", (0, 0, 255)  # Red
    return "Unknown", (255, 255, 255)  # White

# Function to draw bounding boxes on detected objects
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    category, color = get_hazard_category(class_id)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, f"{category} ({confidence:.2f})", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if class_id in movable_objects:
        cv2.putText(img, "Movable", (x - 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Using LLM to present hazard analysis
def analyze_hazard(class_id, direction):
    category, _ = get_hazard_category(class_id)
    prompt = f"An object {yolo_classes[class_id]} classified as '{category}' has been detected in the {direction} direction and is {"" if class_id in movable_objects else "not"} a movable object. This object may pose a risk to a crane depending on its category. Given this information give a brief set of instructions to the crane operator to mitigate the risk. Keep it down to 2 - 3 lines. Please vary the instructions for different objects in the image. Use only third person and use only passive voice. Please maintain the label, hazardous category and movable or not in every image. Dont ask for anything like- provide next detcted object."
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a hazard assessment assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Read image and get its dimensions
image = cv2.imread(args.image)
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet(args.weights, args.config)
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
directions = []
conf_threshold = 0.5
nms_threshold = 0.4

# Get detected objects
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w // 2
            y = center_y - h // 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
            
            # Determine direction based on position in frame
            if center_x < Width // 3:
                direction = "left"
            elif center_x > 2 * Width // 3:
                direction = "right"
            else:
                direction = "center"
            directions.append(direction)

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

print("New Image Detected...")
print()

print(f"Detected {len(indices)} objects in the image.")
print()
with open(output_file, "a") as f:
    f.write(f"Detected {len(indices)} objects in the image.\n\n")

# Display detected objects and hazard analysis (refer hazard_analysis.txt for complete analysis)
j = 0
for i in indices:
    try:
        box = boxes[i]
    except:
        i = i[0]
        box = boxes[i]
    x, y, w, h = box
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    hazard_comment = analyze_hazard(class_ids[i], directions[i])
    print(f"{j + 1}. Hazard Analysis: {hazard_comment}")
    print()
    with open(output_file, "a") as f:
        f.write(f"{j + 1}. Hazard Analysis: {hazard_comment}\n\n")
    j = j + 1
print()

# Display the image with detected objects
cv2.imshow("Object Detection", image)
cv2.waitKey()
resized_image = cv2.resize(image, (800, 600))
cv2.imwrite("object-detection.jpg", resized_image)
cv2.destroyAllWindows()