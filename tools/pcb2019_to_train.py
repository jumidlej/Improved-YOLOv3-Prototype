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

# txt labels path (yolo format)
yolo_labels_path = "/home/jacq/Documentos/Datasets/pcb_dataset/yolo/labels/"

# xml labels path (pascal format)
pascal_labels_path = "/home/jacq/Documentos/Datasets/pcb_dataset/pascal/labels/"

# images path
yolo_images_path = "/home/jacq/Documentos/Datasets/pcb_dataset/yolo/images/"

# data augmentation
data_augmentation = False

# 1. label normalization
xml_files = list_xml_files(path=pascal_labels_path)
normalization(pascal_labels_path, xml_files)

# 2. xml(pascal) to txt(yolo)
images_name = list_images_name(path=yolo_images_path)
components_dict = components_dictionary(classes_file='classes.txt')
pascal_to_yolo_labelImg(yolo_images_path, yolo_labels_path, pascal_labels_path, images_name, components_dict)

# 3. data augmentation
if data_augmentation:
    data_dir = yolo_images_path[:-1]
    label_dir = yolo_labels_path[:-1]

    list_ds = tf.data.Dataset.list_files(str(data_dir+'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(47, reshuffle_each_iteration=False)

    rotate(list_ds, data_dir, label_dir, list_ds)

    list_ds = tf.data.Dataset.list_files(str(data_dir+'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(47, reshuffle_each_iteration=False)

    flip_left_right(list_ds, data_dir, label_dir, list_ds)

    list_ds = tf.data.Dataset.list_files(str(data_dir+'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(47, reshuffle_each_iteration=False)

    flip_up_down(list_ds, data_dir, label_dir, list_ds)

    list_ds = tf.data.Dataset.list_files(str(data_dir+'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(47, reshuffle_each_iteration=False)

    grayscale(list_ds, data_dir, label_dir, list_ds)

    saturation(list_ds, data_dir, label_dir, list_ds)

    brightness(list_ds, data_dir, label_dir, list_ds)

# 4. txt(yolo) to training format
yolo_to_training(yolo_images_path, yolo_labels_path)