import numpy as np
import tensorflow as tf
from yolov3.configs import *
from yolov3.yolov4 import Create_Yolo

def main():
    yolo = Create_Yolo(input_size=416, CLASSES="tools/labels.txt")

    checkpoints_path = "checkpoints/yolov3_custom_Tiny"
    yolo.load_weights(checkpoints_path)

    # yolo.save('save/yolov3')
    yolo.save('save/yolov3.h5')

main()