import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import winsound
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# Load pre-trained Faster R-CNN model for detection
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
# Helper function to calculate distance between bounding box centers
def calculate_distance(box1, box2):
 center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
 center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
 return np.linalg.norm(np.array(center1) - np.array(center2))
# Helper function to draw bounding boxes
def draw_boxes(image, boxes, labels, scores, threshold=0.3):
 image_np = np.array(image)
 for box, label, score in zip(boxes, labels, scores):
 if score >= threshold:
 x1, y1, x2, y2 = box
 cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
 cv2.putText(image_np, f'{label}: {score:.2f}', (int(x1), int(y1)-10),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 return Image.fromarray(image_np)
# Define labels of interest
labels_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
 11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter',
 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse',
 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra',
 24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag',
 28: 'tie', 29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard',
 33: 'sports ball', 34: 'kite', 35: 'baseball bat', 36: 'baseball glove',
25
 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket', 40: 'bottle',
 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon',
 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange',
 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza',
 55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant',
 60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'TV',
 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard',
 68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster',
 72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock',
 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier',
 80: 'toothbrush'}
# Load the image
image_path = r"C:\Users\BAVI\Downloads\img_8.jpg" # Path to your image
image = Image.open(image_path)
# Convert image to tensor
transform = T.Compose([T.ToTensor()])
img_tensor = transform(image)
# Perform object detection
with torch.no_grad():
 prediction = model([img_tensor])
# Parse predictions
boxes = prediction[0]['boxes'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()
# Check distances and generate alert if necessary
people = [1] # Person label
vehicles = [3, 6, 8] # Car, Bus, Truck labels
alert_triggered = False
# Prepare for precision, recall, F1, and accuracy calculation
y_true = [] # Ground truth labels (update based on actual data)
y_pred = []
# Simulate ground truth: Replace with actual ground truth data
# Modify this part according to your dataset
ground_truth_people = 1 # Change according to your ground truth
for i in range(len(boxes)):
 if labels[i] in people and scores[i] > 0.5:
 y_pred.append(1) # Detected person
 else:
 y_pred.append(0) # No person detected
26
# Update y_true based on your ground truth
y_true = [1 if ground_truth_people > 0 else 0] * len(y_pred)
# Calculate alert based on distances
for i, label1 in enumerate(labels):
 if label1 in people and scores[i] > 0.5:
 person_box = boxes[i]
 for j, label2 in enumerate(labels):
 if label2 in vehicles and scores[j] > 0.5:
 vehicle_box = boxes[j]
 distance = calculate_distance(person_box, vehicle_box)
 if distance < 100: # Threshold for "too close" (adjust as needed)
 alert_triggered = True
 break
 if alert_triggered:
 break
# Generate alert if a person is too close to a vehicle
if alert_triggered:
 print("Warning: Person is too close to a vehicle!")
 winsound.Beep(1000, 5000) # Frequency: 1000 Hz, Duration: 5000 milliseconds (5
seconds)
# Draw bounding boxes on the image
image_with_boxes = draw_boxes(image, boxes, [labels_map.get(label, 'unknown') for
label in labels], scores)
# Display the image with bounding boxes
plt.imshow(image_with_boxes)
plt.axis('off')
plt.show()
# Calculate precision, recall, F1 score, and accuracy
if len(y_true) == len(y_pred):
 precision = precision_score(y_true, y_pred, zero_division=0)
 recall = recall_score(y_true, y_pred, zero_division=0)
 f1 = f1_score(y_true, y_pred, zero_division=0)
 accuracy = accuracy_score(y_true, y_pred)
 print(f"Precision: {precision:.2f}")
 print(f"Recall: {recall:.2f}")
 print(f"F1 Score: {f1:.2f}")
 print(f"Accuracy: {accuracy:.2f}")
else:
 print("Length mismatch between y_true and y_pred. Please check your inputs.");