import cv2
from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import torch
import cv2 
import numpy as np
import pathlib
import matplotlib.pyplot as plt



model = YOLO("D:/OneDrive/OneDrive - Student Ambassadors/College notes/Sem 7/github/New folder/atomizen-testing/best_human5s.pt")
model.predict(source="0", show=True, conf=0.5)

