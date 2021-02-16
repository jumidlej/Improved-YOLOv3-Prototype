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
    image_path = "/home/jacq/Documentos/segmentation/images/opencv_frame_01.png"
    results_path = "images/"

    extension = image_path.split(".")[1]
    image_name = image_path.split(".")[0]
    image_name = image_name.split("/")[-1]
    image_path = image_path[:len(image_path)-(len(image_name)+len(extension)+1)]
    # print(image_path)
    # print(image_name)
    # print(extension)

    segment_pcbs(image_path+image_name+"."+extension, results_path)

    pcb_final_cut(results_path+image_name+"_5."+extension, results_path)

    pcb_final_cut(results_path+image_name+"_6."+extension, results_path)

    yolo = Load_Yolo_model()
    detect_image(yolo, results_path+image_name+"_5_11."+extension, results_path+image_name+"_5_12."+extension, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES)

    yolo = Load_Yolo_model()
    detect_image(yolo, results_path+image_name+"_6_11."+extension, results_path+image_name+"_6_12."+extension, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES)

main()