import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, Load_Yolo_model
from yolov3.configs import *
from segmentation.pcbs_perspective import * 
from segmentation.segment_pcbs import *

def main():
    image_path = "/home/pi/image/"
    image_name = list_png_files(image_path)
    extension = "png"
    results_path = "/home/pi/results/"

    segment_pcbs(image_path+image_name+"."+extension, results_path)

    pcb_final_cut(results_path+image_name+"_5."+extension, results_path)

    pcb_final_cut(results_path+image_name+"_6."+extension, results_path)

    yolo = Load_Yolo_model()
    detect_image(yolo, results_path+image_name+"_5_11."+extension, results_path+image_name+"_5_12."+extension, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES)

    yolo = Load_Yolo_model()
    detect_image(yolo, results_path+image_name+"_6_11."+extension, results_path+image_name+"_6_12."+extension, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES)

main()