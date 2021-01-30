import cv2
import xml.etree.ElementTree as ET
from os import listdir

yolo_labels_path = "/home/jacq/Documentos/Datasets/pcb_augmented/labels/"

yolo_images_path = "/home/jacq/Documentos/Datasets/pcb_augmented/images/"

'''
Does: A .txt file with every image in the yolo_images_path in this format:
    image xmin,ymin,xmax,ymax,class_number xmin,ymin,xmax,ymax,class_number ...
'''
def yolo_augmented_to_training():
    images_name = []
    files = [f for f in listdir(yolo_labels_path)]
    for f in files:
        if f[len(f)-4:] == ".txt":
            images_name.append(f[:-4])

    train_file = open("train_augmented.txt", 'w')
    for name in images_name:
        train_file.write(yolo_images_path+name+".jpg")
        image = cv2.imread(yolo_images_path+name+".jpg")

        labels_file = open(yolo_labels_path+name+".txt", "r")

        for line in labels_file:
            line = line.split()

            x = float(line[1])
            y = float(line[2])
            w = float(line[3])
            h = float(line[4])

            px = image.shape[1]
            py = image.shape[0]

            xmin = int((x-w/2)*px)
            ymin = int((y-h/2)*py)
            xmax = int((x+w/2)*px)
            ymax = int((y+h/2)*py)

            train_file.write(" "+str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)+","+line[0])
        train_file.write("\n")
    train_file.close()

'''
Does: Test if the bounding boxes are correct in the training augmented file
Problem: Basicly never stops so you will have to kill it
'''
def test_yolo_augmented_to_training():
    train = open("train_augmented.txt", "r")
    for line in train:
        line = line.split()
        image = cv2.imread(line[0])
        for i in range(1, len(line)):
            line[i]=line[i].split(",")
            xmin = int(line[i][0])
            ymin = int(line[i][1])
            xmax = int(line[i][2])
            ymax = int(line[i][3])
            image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0), 2)
        
        scale_percent = 60 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Imagem", resized)
        cv2.waitKey(0)

yolo_augmented_to_training()