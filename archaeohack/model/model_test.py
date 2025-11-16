import torch
from ultralytics import YOLO
import os

# Load your trained model
model = YOLO("/Users/khalonmanswell/Documents/GitHub/ArchaeoHack-Group-Still-Loading/runs/classify/17594/weights/best.pt")
image_path = "/Users/khalonmanswell/Documents/GitHub/ArchaeoHack-Group-Still-Loading/archaeohack/data/utf-pngs/p4.png"

results = model.predict(image_path)
print(results[0])
