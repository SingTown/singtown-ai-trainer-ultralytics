"""Export trained weights to an INT8 TFLite model. Run by main.py."""

import sys

from ultralytics import YOLO

weights, height, width = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

YOLO(weights).export(
    format="tflite",
    int8=True,
    data="../dataset/data.yaml",
    simplify=True,
    end2end=False,
    imgsz=(height, width),
)
