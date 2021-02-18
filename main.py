import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, Load_Yolo_model
from yolov3.configs import *
from segmentation.pcbs_perspective import * 
from segmentation.segment_pcbs import *

def list_files(path=None):
    if path == None:
        print("Nenhuma pasta foi especificada.")
        return 0

    images = []
    files = [f for f in listdir(path)]
    for f in files:
        images.append(f)

    return images

def main():
    # "/home/jacq/Documentos/datasets/C920/"
    # "/home/jacq/"
    image_path = "/home/pi/image/"
    results_path = "/home/pi/results/"

    images = list_files(image_path)
    image_name = images[0].split(".")[0]
    extension = images[0].split(".")[1]

    os.mkdir(results_path+image_name, mode=0o755)
    results_path = results_path+image_name+"/"

    image_1, image_2 = segment_pcbs(image_path+image_name+"."+extension, results_path)

    pcb_1 = pcb_final_cut(image_1, results_path)

    pcb_2 = pcb_final_cut(image_2, results_path)

    yolo = Load_Yolo_model()
    image_1, bboxes_1 = detect_image(yolo, pcb_1, results_path+image_name+"_detection_1."+extension, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES)

    yolo = Load_Yolo_model()
    image_2, bboxes_2 = detect_image(yolo, pcb_2, results_path+image_name+"_detection_2."+extension, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES)

    txt_1 = open(results_path+image_name+"_detection_1.txt", "w")
    txt_2 = open(results_path+image_name+"_detection_2.txt", "w")

    for bbox in bboxes_1:
        txt_1.write(str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3])+" "+str(bbox[4])+" "+str(bbox[5])+"\n")
    for bbox in bboxes_2:
        txt_2.write(str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3])+" "+str(bbox[4])+" "+str(bbox[5])+"\n")

    txt_1.close()
    txt_2.close()

main()