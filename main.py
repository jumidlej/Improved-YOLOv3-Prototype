import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *
from segmentation.pcbs_perspective import * 
from segmentation.segment_pcbs import *

def main():
    image_path = "/home/jacq/Documentos/segmentation/images/opencv_frame_01.png"
    results_path = "images/"

    segment_pcbs(image_path, results_path)

    image_path = "images/opencv_frame_01_5.png"

    pcb_final_cut(image_path, results_path)

    image_path = "images/opencv_frame_01_6.png"

    pcb_final_cut(image_path, results_path)

    image_path   = "images/opencv_frame_01_5_11.png"
    results_path = "images/opencv_frame_01_5_12.png"

    yolo = Load_Yolo_model()
    detect_image(yolo, image_path, results_path, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES)

    image_path   = "images/opencv_frame_01_6_11.png"
    results_path = "images/opencv_frame_01_6_12.png"

    yolo = Load_Yolo_model()
    detect_image(yolo, image_path, results_path, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES)

main()