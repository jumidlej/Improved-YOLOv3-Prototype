import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import listdir
from label_normalization import *
from xml_to_yolo import *
from yolo_to_training import *
from traditional import *

# SETUP
# txt labels path (yolo format)
yolo_labels_path = "/home/jacq/Documentos/Datasets/pcb_yolo/labels/"

# xml labels path (pascal format)
pascal_labels_path = "/home/jacq/Documentos/Datasets/pcb_pascal/labels/"

# images path
yolo_images_path = "/home/jacq/Documentos/Datasets/pcb_yolo/images/"

# data augmentation
data_augmentation = True

# excluded and repeated from the classes.txt file
excluded = ["text"]
repeated = []

# training file name
training_file_name = "training_file.txt"

# CODE
# 1. label normalization
normalization(pascal_labels_path)

# 2. xml(pascal) to txt(yolo)
images_names = list_images_names(path=yolo_images_path)
components_dict = components_dictionary("classes.txt", excluded, repeated)
pascal_to_yolo_labelImg(yolo_images_path, yolo_labels_path, pascal_labels_path, images_names, components_dict, excluded)

# 3. data augmentation
if data_augmentation:
    data_dir = yolo_images_path[:-1]
    label_dir = yolo_labels_path[:-1]

    list_ds = tf.data.Dataset.list_files(str(data_dir+'*/*'), shuffle=False)

    rotate(list_ds, data_dir, label_dir, list_ds)

    list_ds = tf.data.Dataset.list_files(str(data_dir+'*/*'), shuffle=False)

    flip_left_right(list_ds, data_dir, label_dir, list_ds)

    list_ds = tf.data.Dataset.list_files(str(data_dir+'*/*'), shuffle=False)

    flip_up_down(list_ds, data_dir, label_dir, list_ds)

    list_ds = tf.data.Dataset.list_files(str(data_dir+'*/*'), shuffle=False)

    grayscale(list_ds, data_dir, label_dir, list_ds)

    saturation(list_ds, data_dir, label_dir, list_ds)

    brightness(list_ds, data_dir, label_dir, list_ds)

# 4. txt(yolo) to training format
yolo_to_training(training_file_name, yolo_images_path, yolo_labels_path)