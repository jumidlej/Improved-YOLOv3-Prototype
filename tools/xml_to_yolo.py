import cv2
import xml.etree.ElementTree as ET
from os import listdir

excluded = ["text", "component text"]
repeated = ["capacitor jumper", "resistor jumper", "resistor network", "diode zener array"]

'''
Argument: Images path (.jpg)
Returns: List of images name
'''
def list_images_name(path=None):
    if path == None:
        print("Nenhuma pasta foi especificada.")
        return 0

    images = []
    files = [f for f in listdir(path)]
    for f in files:
        if f[len(f)-4:] == ".jpg":
            images.append(f[:-4])

    return images

'''
Argument: A classes.txt file
Does: A labels.txt file with all components that are not excluded or repeated
Returns: A components dictionary in which components have a respective number
'''
def components_dictionary(classes_file):
    components_dict = {}

    classes = open(classes_file, "r")
    labels = open("labels.txt", "w")
    component_number = 0
    for component in classes:
        if component[:-1] not in excluded:
            if component[:-1] in repeated:
                component_number -= 1
                components_dict[component[:-1]] = component_number
            else:
                components_dict[component[:-1]] = component_number
                labels.write(component)
            component_number += 1

    labels.close()
    classes.close()

    return components_dict

'''
Arguments: List of images name, dictionary with components and its numbers
Does: A .txt file to every image in yolo format
'''
def pascal_to_yolo_labelImg(yolo_images_path, yolo_labels_path, pascal_labels_path, images_name, components_dict):
    for name in images_name:
        image = cv2.imread(yolo_images_path+name+".jpg")
        yolo = open(yolo_labels_path+name+".txt", 'w')

        tree = ET.parse(pascal_labels_path+name+".xml")
        root = tree.getroot()

        for child in root.findall("object"):
            component_name = child[0].text

            quotes = component_name.split('"')
            if len(quotes)>=3 and quotes[0]=="connector " and quotes[1].split()[0]=="Port":
                component_name = "connector Port"
                # print(component_name)
            elif len(quotes)>=3 and component_name[0]=='"':
                component_name = quotes[1]
            else:
                component_name = component_name.split()[0]

            xmin = int(child[4][0].text)
            ymin = int(child[4][1].text)
            xmax = int(child[4][2].text)
            ymax = int(child[4][3].text)

            #print(image.shape)
            px = image.shape[1]
            py = image.shape[0]

            # cálculo pro quadradinho ser igual o do yolo
            # ponto no centro, altura e largura, dividos pelo tamanho da imagem
            X = ((xmax+xmin)/2)/px
            Y = ((ymax+ymin)/2)/py
            W = (xmax-xmin)/px
            H = (ymax-ymin)/py

            if component_name not in excluded:
                yolo.write(str(components_dict[component_name])+" "+str(X)+" "+str(Y)+" "+str(W)+" "+str(H)+"\n")

        yolo.close()
