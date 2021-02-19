import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, Load_Yolo_model
from yolov3.configs import *
from segmentation.pcbs_perspective import * 
from segmentation.segment_pcbs import *
from threading import Thread

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

    try:
        os.mkdir(results_path+image_name)
    except:
        print("Path "+results_path+image_name+" already exists")
    finally:
        results_path = results_path+image_name+"/"

    image_1, image_2 = segment_pcbs(image_path+image_name+"."+extension, results_path)

    yolo = Load_Yolo_model()

    bboxes = {}
    images = {}

    def align_and_detect(index, image):
        pcb = pcb_final_cut(image, None)
        images[index] = pcb
        bboxes[index] = detect_image(yolo, pcb, input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)

    # print("Split")
    thread_1 = Thread(target=align_and_detect,args=['1', image_1])
    thread_2 = Thread(target=align_and_detect,args=['2', image_2])

    thread_1.start()
    # print("Start thread 1")
    thread_2.start()
    # print("Start thread 2")

    thread_1.join()
    # print("Join thread 1")
    thread_2.join()
    # print("Join thread 2")

    # print("Continue")

    for index in range(1, 3):
        txt = open(results_path+image_name+"_detection_"+str(index)+".txt", "w")

        for bbox in bboxes[str(index)]:
            txt.write(str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3])+" "+str(bbox[4])+" "+str(bbox[5])+"\n")

            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])

            if class_ind == 0:
                rectangle_colors=(255, 255, 0)
            elif class_ind == 1:
                rectangle_colors=(155, 0, 255)
            elif class_ind == 2:
                rectangle_colors=(0, 255, 255)
            elif class_ind == 3:
                rectangle_colors=(0, 0, 0)
            elif class_ind == 4:
                rectangle_colors=(255, 255, 255)

            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(images[str(index)], (x1, y1), (x2, y2), rectangle_colors, 1)
            cv2.imwrite(results_path+image_name+"_detection_"+str(index)+"."+extension, images[str(index)])

        txt.close()
        index += 1


main()