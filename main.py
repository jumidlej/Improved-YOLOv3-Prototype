import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image
from yolov3.configs import *
from yolov3.yolov4 import Create_Yolo
from segmentation.pcbs_perspective import * 
from segmentation.segment_pcbs import *
from threading import Thread
from datetime import datetime

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
    begin = datetime.now()
    print("Started =", begin)

    image_path = "/home/pi/image/"
    results_path = "/home/pi/results/"
    checkpoints_path = "checkpoints/yolov3_custom_Tiny"

    images = list_files(image_path)
    image_name = images[0].split(".")[0]
    extension = images[0].split(".")[1]

    try:
        os.mkdir(results_path+image_name)
    except:
        print("Path "+results_path+image_name+" already exists")
    finally:
        results_path = results_path+image_name+"/"

    segment_time = datetime.now()
    print("Make dir time =", segment_time-begin)
    image_1, image_2 = segment_pcbs(image_path+image_name+"."+extension, results_path)

    create_yolo_time = datetime.now()
    print("Segment time =", create_yolo_time-segment_time)
    yolo = Create_Yolo(input_size=416, CLASSES="tools/labels.txt")

    load_weights_time = datetime.now()
    print("Create yolo time =", load_weights_time-create_yolo_time)
    yolo.load_weights(checkpoints_path)

    # load_yolo_time = datetime.now()
    # print("Load weights time =", load_yolo_time-load_weights_time)
    # yolo_2 = tf.keras.models.load_model('save/yolov3', compile=False)

    # load_yolo_time_h5 = datetime.now()
    # print("Load model yolo time =", load_yolo_time_h5-segment_time)
    # yolo = tf.keras.models.load_model('save/yolov3.h5', compile=False)

    bboxes = {}
    images = {}

    def align_and_detect(index, image):
        aling_time = datetime.now()
        print("Load model yolo h5 time "+index+" =", aling_time-load_weights_time)
        pcb = pcb_final_cut(image, None)

        images[index] = pcb

        detect_time = datetime.now()
        print("Align time "+index+" =", detect_time-aling_time)
        print("Detect start "+index+" =", detect_time)
        bboxes[index] = detect_image(yolo, pcb, input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)

    # print("Split")
    thread_1 = Thread(target=align_and_detect,args=['1', image_1])
    thread_2 = Thread(target=align_and_detect,args=['2', image_2])

    thread_1.start()
    print("Start thread 1")
    thread_2.start()
    print("Start thread 2")

    thread_1.join()
    print("Join thread 1")
    thread_2.join()
    print("Join thread 2")

    print("Continue")

    write_time = datetime.now()
    print("Detect end =", write_time)
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

    end_time = datetime.now()
    print("Write time =", end_time-write_time)

main()